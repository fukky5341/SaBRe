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
execution time: IAR + RelationalAnalysis = 23.64 + 36.96 = 60.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.4952888, upper bound: 1.4952880

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930242, upper bound: 1.4952809
time: 9.68 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952817, upper bound: 1.4952813
time: 6.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 16.54 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 16.54
Output dim: 8, lower bound: -1.4930242, upper bound: 1.4952809
NS_A2, status: Status.UNKNOWN, split count: 1, time: 16.54
Output dim: 8, lower bound: -1.4952817, upper bound: 1.4952813

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -11.0289764, -6.8591242, -11.0299339, -6.8570595, -3.1110620, 3.1093912
1: -9.9610510, -6.7885375, -9.9668093, -6.7874899, -2.7875094, 2.7918410
2: -4.8398786, -1.7017742, -4.8408098, -1.6950128, -2.8390412, 2.8322558
3: -1.7212691, 1.6580504, -1.7237175, 1.6607997, -3.3820689, 3.3817677
4: -13.9953794, -10.0299644, -14.0008450, -10.0292311, -3.2779074, 3.2829895
5: -8.5511675, -5.0987415, -8.5538712, -5.0979543, -2.1599278, 2.1619563
6: -12.7698059, -8.5451288, -12.7711954, -8.5420656, -3.2905378, 3.2889867
7: -9.1839571, -5.7023139, -9.1901073, -5.7015314, -2.8337564, 2.8390965
8: 9.6516809, 12.5805035, 9.6456785, 12.5810013, -2.6667404, 2.6733260
9: -7.9719172, -3.7013092, -7.9725308, -3.7004178, -2.9964933, 2.9957151

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930219, upper bound: 1.4918503
time: 6.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930217, upper bound: 1.4952783
time: 9.32 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -11.0377007, -6.8517323, -11.0312004, -6.8542795, -3.1243005, 3.1244550
1: -9.9825945, -6.7680578, -9.9745502, -6.7860956, -2.8081861, 2.8194928
2: -4.8699255, -1.6805713, -4.8420515, -1.6859416, -2.8718596, 2.8484478
3: -1.7423849, 1.6700649, -1.7269833, 1.6644989, -3.4068837, 3.3970482
4: -14.0116749, -10.0041485, -14.0081930, -10.0282612, -3.2936316, 3.3167562
5: -8.5616503, -5.0880117, -8.5575132, -5.0969095, -2.1693201, 2.1761413
6: -12.7881203, -8.5354910, -12.7730389, -8.5379362, -3.3128366, 3.2967787
7: -9.2025146, -5.6786852, -9.1983671, -5.7004972, -2.8473616, 2.8710012
8: 9.6287956, 12.5982714, 9.6376400, 12.5816555, -2.6898413, 2.7011070
9: -7.9776435, -3.6958714, -7.9733362, -3.6992133, -3.0041800, 3.0030971

Time for backsubstitution: 21.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952792, upper bound: 1.4918478
time: 8.38 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952792, upper bound: 1.4952789
time: 6.42 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.42 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.42
Output dim: 8, lower bound: -1.4930219, upper bound: 1.4918503
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.42
Output dim: 8, lower bound: -1.4930217, upper bound: 1.4952783
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.42
Output dim: 8, lower bound: -1.4952792, upper bound: 1.4918478
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.42
Output dim: 8, lower bound: -1.4952792, upper bound: 1.4952789

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -11.0238581, -6.8827534, -10.9930849, -6.8987536, -3.0657377, 3.0520248
1: -9.9572430, -6.7927899, -9.9423857, -6.7953248, -2.7727747, 2.7607174
2: -4.8341970, -1.7054825, -4.8268914, -1.7089310, -2.8188910, 2.8131952
3: -1.6998501, 1.6517205, -1.6850262, 1.6204880, -3.3203382, 3.3367467
4: -13.9887590, -10.0505257, -13.9621506, -10.0650969, -3.2369556, 3.2270451
5: -8.5486135, -5.1099243, -8.5343094, -5.1186686, -2.1362739, 2.1304934
6: -12.7667665, -8.5504627, -12.7603817, -8.5537148, -3.2715678, 3.2686586
7: -9.1751575, -5.7052851, -9.1709099, -5.7064886, -2.8140078, 2.8157611
8: 9.6598644, 12.5777187, 9.6634293, 12.5640783, -2.6402068, 2.6507893
9: -7.9650545, -3.7081106, -7.9499083, -3.7145319, -2.9749055, 2.9651318

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897179, upper bound: 1.4906117
time: 7.06 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930155, upper bound: 1.4918443
time: 6.66 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -11.0289707, -6.8591433, -11.0299282, -6.8570867, -3.0749035, 3.1093650
1: -9.9610472, -6.7885423, -9.9668064, -6.7874942, -2.7851248, 2.7918305
2: -4.8398724, -1.7017770, -4.8408003, -1.6950186, -2.8390298, 2.8277097
3: -1.7212484, 1.6580453, -1.7236836, 1.6607878, -3.3820362, 3.3817289
4: -13.9953718, -10.0299759, -14.0008335, -10.0292549, -3.2684126, 3.2829609
5: -8.5511637, -5.0987515, -8.5538635, -5.0979733, -2.1399312, 2.1619422
6: -12.7698030, -8.5451374, -12.7711906, -8.5420732, -3.2965398, 3.2778215
7: -9.1839485, -5.7023187, -9.1900921, -5.7015352, -2.8215799, 2.8486056
8: 9.6516876, 12.5805016, 9.6456881, 12.5809956, -2.6667275, 2.6648304
9: -7.9719100, -3.7013183, -7.9725184, -3.7004280, -2.9952984, 2.9988713

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897177, upper bound: 1.4940419
time: 7.94 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930153, upper bound: 1.4952723
time: 6.18 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -11.0325775, -6.8753538, -10.9943514, -6.8959694, -3.0789752, 3.0670829
1: -9.9787674, -6.7723107, -9.9501400, -6.7939305, -2.7934289, 2.7871852
2: -4.8642464, -1.6842808, -4.8281307, -1.6998566, -2.8510122, 2.8293834
3: -1.7209888, 1.6637623, -1.6883099, 1.6241982, -3.3451869, 3.3520722
4: -14.0050764, -10.0247087, -13.9695101, -10.0641222, -3.2526426, 3.2607937
5: -8.5591040, -5.0991917, -8.5379553, -5.1176271, -2.1456509, 2.1446900
6: -12.7850666, -8.5408173, -12.7622185, -8.5495806, -3.2938447, 3.2764401
7: -9.1937227, -5.6816602, -9.1791821, -5.7054539, -2.8275719, 2.8476410
8: 9.6369743, 12.5954819, 9.6553841, 12.5647345, -2.6633096, 2.6785660
9: -7.9707499, -3.7026708, -7.9507079, -3.7133226, -2.9825583, 2.9725146

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919752, upper bound: 1.4906139
time: 6.36 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952729, upper bound: 1.4918420
time: 9.37 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -11.0376949, -6.8517466, -11.0311947, -6.8543072, -3.0881433, 3.1244283
1: -9.9825907, -6.7680626, -9.9745445, -6.7861032, -2.8058023, 2.8131423
2: -4.8699179, -1.6805737, -4.8420444, -1.6859484, -2.8679571, 2.8439016
3: -1.7423673, 1.6700577, -1.7269542, 1.6644881, -3.4068553, 3.3970118
4: -14.0116673, -10.0041599, -14.0081816, -10.0282831, -3.2841330, 3.3026543
5: -8.5616493, -5.0880218, -8.5575123, -5.0969300, -2.1493230, 2.1730080
6: -12.7881165, -8.5354939, -12.7730360, -8.5379467, -3.3188219, 3.2856331
7: -9.2025051, -5.6786857, -9.1983528, -5.7005000, -2.8352127, 2.8805118
8: 9.6287994, 12.5982695, 9.6376524, 12.5816526, -2.6898265, 2.6926103
9: -7.9776359, -3.6958797, -7.9733253, -3.6992211, -3.0029845, 3.0062528

Time for backsubstitution: 23.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919752, upper bound: 1.4940419
time: 7.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952729, upper bound: 1.4952723
time: 5.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 36.98 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4897179, upper bound: 1.4906117
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4930155, upper bound: 1.4918443
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4897177, upper bound: 1.4940419
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4930153, upper bound: 1.4952723
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4919752, upper bound: 1.4906139
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4952729, upper bound: 1.4918420
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4919752, upper bound: 1.4940419
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 36.98
Output dim: 8, lower bound: -1.4952729, upper bound: 1.4952723

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -10.9895029, -6.8982134, -10.9727221, -6.9076428, -3.0109091, 3.0063324
1: -9.9485550, -6.8122463, -9.9373140, -6.8068404, -2.7508211, 2.7345304
2: -4.8293657, -1.7211998, -4.8240604, -1.7182561, -2.7961035, 2.7876034
3: -1.6854148, 1.6462164, -1.6764755, 1.6171966, -3.3026114, 3.3226919
4: -13.9811325, -10.0658398, -13.9576855, -10.0741997, -3.2143278, 3.2019620
5: -8.5379381, -5.1131368, -8.5279894, -5.1205549, -2.1220164, 2.1195869
6: -12.7529869, -8.5655174, -12.7519398, -8.5626869, -3.2472892, 3.2441325
7: -9.1670361, -5.7295017, -9.1662178, -5.7208118, -2.7879276, 2.7832289
8: 9.6796188, 12.5745592, 9.6751156, 12.5622253, -2.6160812, 2.6324039
9: -7.9522772, -3.7139187, -7.9423242, -3.7179368, -2.9570003, 2.9499540

Time for backsubstitution: 23.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4872948, upper bound: 1.4900388
time: 10.28 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897155, upper bound: 1.4906096
time: 7.75 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -11.0282393, -6.8498993, -10.9930391, -6.8987651, -3.0615401, 3.0895061
1: -9.9863892, -6.7899356, -9.9423771, -6.7953520, -2.7989488, 2.7600069
2: -4.8499489, -1.7024683, -4.8268871, -1.7089541, -2.8290596, 2.8140121
3: -1.7169545, 1.6617643, -1.6850076, 1.6204838, -3.3374381, 3.3467717
4: -14.0018606, -10.0408506, -13.9621429, -10.0651140, -3.2455864, 3.2323828
5: -8.5529156, -5.0994787, -8.5342970, -5.1186719, -2.1397610, 2.1402869
6: -12.7795258, -8.5427523, -12.7603674, -8.5537357, -3.2842665, 3.2750916
7: -9.2038517, -5.6990800, -9.1709051, -5.7065191, -2.8401384, 2.8170280
8: 9.6437187, 12.5885468, 9.6634531, 12.5640764, -2.6513596, 2.6603289
9: -7.9703860, -3.6946392, -7.9498925, -3.7145352, -2.9808683, 2.9797211

Time for backsubstitution: 23.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4905900, upper bound: 1.4912664
time: 7.88 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930131, upper bound: 1.4918396
time: 22.19 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.9946327, -6.8746085, -11.0095997, -6.8659778, -3.0200663, 3.0636501
1: -9.9523726, -6.8079939, -9.9617500, -6.7990098, -2.7631712, 2.7656527
2: -4.8350286, -1.7175057, -4.8379626, -1.7043576, -2.8162556, 2.8021479
3: -1.7068772, 1.6525506, -1.7151673, 1.6575232, -3.3644004, 3.3614326
4: -13.9877100, -10.0452099, -13.9963522, -10.0383148, -3.2457781, 3.2578554
5: -8.5404892, -5.1019688, -8.5475407, -5.0998693, -2.1256614, 2.1510210
6: -12.7560024, -8.5601683, -12.7627029, -8.5510254, -3.2723627, 3.2533212
7: -9.1757765, -5.7265358, -9.1853466, -5.7158651, -2.7954264, 2.8160014
8: 9.6714363, 12.5773363, 9.6573715, 12.5791435, -2.6426177, 2.6464584
9: -7.9591293, -3.7071235, -7.9649363, -3.7038224, -2.9773684, 2.9836497

Time for backsubstitution: 22.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891315, upper bound: 1.4890441
time: 8.54 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891315, upper bound: 1.4934580
time: 8.39 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -11.0333672, -6.8262701, -11.0298843, -6.8570986, -3.0707331, 3.1294222
1: -9.9901876, -6.7856874, -9.9667950, -6.7875247, -2.8075829, 2.7911172
2: -4.8556552, -1.6987650, -4.8407965, -1.6950412, -2.8492789, 2.8285170
3: -1.7383628, 1.6680551, -1.7236629, 1.6607845, -3.3991473, 3.3905869
4: -14.0084772, -10.0202150, -14.0008268, -10.0292721, -3.2771072, 3.2883005
5: -8.5554552, -5.0883064, -8.5538511, -5.0979795, -2.1433878, 2.1698825
6: -12.7825642, -8.5374241, -12.7711773, -8.5420961, -3.3092594, 3.2844019
7: -9.2127523, -5.6961093, -9.1900845, -5.7015667, -2.8479862, 2.8498912
8: 9.6355543, 12.5913258, 9.6457138, 12.5809917, -2.6778579, 2.6743836
9: -7.9772329, -3.6878181, -7.9725018, -3.7004325, -3.0012245, 3.0134845

Time for backsubstitution: 22.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924314, upper bound: 1.4902768
time: 7.55 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924315, upper bound: 1.4946891
time: 5.93 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.9982214, -6.8908153, -10.9739885, -6.9048600, -3.0241890, 3.0212164
1: -9.9700832, -6.7917767, -9.9450684, -6.8054490, -2.7714515, 2.7610514
2: -4.8594074, -1.6999673, -4.8252983, -1.7091742, -2.8265896, 2.8037834
3: -1.7064633, 1.6582701, -1.6797457, 1.6209090, -3.3273723, 3.3380158
4: -13.9974775, -10.0400229, -13.9650497, -10.0732241, -3.2299833, 3.2357044
5: -8.5484505, -5.1023998, -8.5316353, -5.1195145, -2.1313615, 2.1337881
6: -12.7711678, -8.5558605, -12.7537861, -8.5585537, -3.2695208, 3.2519236
7: -9.1855774, -5.7058868, -9.1744909, -5.7197771, -2.8014717, 2.8150883
8: 9.6566706, 12.5923204, 9.6670618, 12.5628805, -2.6391335, 2.6601882
9: -7.9579716, -3.7084589, -7.9431257, -3.7167268, -2.9646521, 2.9573298

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4895521, upper bound: 1.4900385
time: 5.62 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919728, upper bound: 1.4906096
time: 7.46 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -11.0369663, -6.8424830, -10.9943085, -6.8959799, -3.0747833, 3.0995750
1: -10.0079117, -6.7694411, -9.9501314, -6.7939582, -2.8198218, 2.7865591
2: -4.8800120, -1.6812521, -4.8281250, -1.6998796, -2.8536658, 2.8301649
3: -1.7380760, 1.6737659, -1.6882901, 1.6241958, -3.3622718, 3.3620560
4: -14.0181665, -10.0150280, -13.9695063, -10.0641394, -3.2613068, 3.2661219
5: -8.5634212, -5.0887527, -8.5379391, -5.1176319, -2.1491361, 2.1544819
6: -12.7977962, -8.5331230, -12.7622070, -8.5496016, -3.3065205, 3.2828684
7: -9.2223959, -5.6754608, -9.1791754, -5.7054873, -2.8537560, 2.8488979
8: 9.6207848, 12.6063213, 9.6554079, 12.5647297, -2.6744299, 2.6872928
9: -7.9760904, -3.6892066, -7.9506912, -3.7133317, -2.9885321, 2.9870949

Time for backsubstitution: 22.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4928461, upper bound: 1.4912669
time: 8.05 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952705, upper bound: 1.4918400
time: 7.98 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -11.0033531, -6.8672152, -11.0108671, -6.8631997, -3.0333500, 3.0750551
1: -9.9739208, -6.7875261, -9.9694901, -6.7976179, -2.7838268, 2.7870026
2: -4.8650675, -1.6962690, -4.8392015, -1.6952794, -2.8435287, 2.8183346
3: -1.7279258, 1.6645777, -1.7184269, 1.6612225, -3.3891482, 3.3751221
4: -14.0040321, -10.0193863, -14.0037003, -10.0373392, -3.2614613, 3.2772312
5: -8.5509987, -5.0912371, -8.5511923, -5.0988245, -2.1350226, 2.1612234
6: -12.7742014, -8.5505152, -12.7645531, -8.5468969, -3.2946033, 3.2611456
7: -9.1943102, -5.7029190, -9.1936092, -5.7148294, -2.8090343, 2.8478861
8: 9.6484928, 12.5951042, 9.6493244, 12.5798006, -2.6656694, 2.6742425
9: -7.9648342, -3.7016666, -7.9657440, -3.7026172, -2.9850340, 2.9910264

Time for backsubstitution: 22.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913910, upper bound: 1.4890431
time: 6.12 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913909, upper bound: 1.4934574
time: 5.57 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 60.60 + 540.74 = 601.33 seconds
