## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.487812356
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0312042, -6.8542714, -11.0312042, -6.8542714, -4.1769328, 4.1769328)
1: (-9.9745617, -6.7860947, -9.9745617, -6.7860947, -3.1884670, 3.1884670)
2: (-4.8420544, -1.6859304, -4.8420544, -1.6859304, -3.1561241, 3.1561241)
3: (-1.7269878, 1.6645033, -1.7269878, 1.6645033, -3.3914912, 3.3914912)
4: (-14.0081997, -10.0282602, -14.0081997, -10.0282602, -3.9799395, 3.9799395)
5: (-8.5575237, -5.0969090, -8.5575237, -5.0969090, -3.2181473, 3.2181475)
6: (-12.7730389, -8.5379305, -12.7730389, -8.5379305, -4.2351084, 4.2351084)
7: (-9.1983776, -5.7004948, -9.1983776, -5.7004948, -3.4978828, 3.4978828)
8: (9.6376209, 12.5816565, 9.6376209, 12.5816565, -2.9440355, 2.9440355)
9: (-7.9733381, -3.6992102, -7.9733381, -3.6992102, -4.0845170, 4.0845165)

## BASE Result
execution time: IAR + LP analysis = 15.07 + 35.04 = 50.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4068807, upper bound: 2.4068790


# Binary Search by BASE starts (time budget: 3549.89 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.9311227798461914
rel_dist={8: [-1.8610607868461848, 1.8610605376350104]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.6844100952148438
rel_dist={8: [-1.4952889722192388, 1.4952881994988214]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.519934892654419
rel_dist={8: [-1.2056374419199969, 1.2056392133504037]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.602172374725342
rel_dist={8: [-1.3553178896207374, 1.355318405775778]}

## Binary Search Result
Binary search time: 218.49 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3331.40 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5845
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9688919
time: 4.90 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688918, upper bound: 1.9688919
time: 4.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.99 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.99
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9688919
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.99
Output dim: 8, lower bound: -1.9688918, upper bound: 1.9688919

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -11.0289745, -6.8591261, -11.0307789, -6.8552113, -3.6449161, 3.6417704
1: -9.9610510, -6.7885365, -9.9719582, -6.7865615, -3.1293411, 3.1375608
2: -4.8398800, -1.7017744, -4.8416367, -1.6889775, -3.1509025, 3.1398623
3: -1.7212706, 1.6580515, -1.7258921, 1.6632587, -3.3845291, 3.3839436
4: -13.9953804, -10.0299664, -14.0057287, -10.0285845, -3.7626600, 3.7722468
5: -8.5511646, -5.0987401, -8.5562954, -5.0972600, -2.6276340, 2.6314650
6: -12.7698078, -8.5451317, -12.7724228, -8.5393200, -3.7811728, 3.7782226
7: -9.1839590, -5.7023129, -9.1955996, -5.7008419, -3.3013096, 3.3113842
8: 9.6516819, 12.5805035, 9.6403246, 12.5814390, -2.9297571, 2.9401789
9: -7.9719172, -3.7013087, -7.9730654, -3.6996164, -3.4798412, 3.4783549

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9632779
time: 4.98 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9688918
time: 4.99 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.0377007, -6.8517303, -11.0312042, -6.8542743, -3.6549740, 3.6576114
1: -9.9825935, -6.7680578, -9.9745569, -6.7860951, -3.1507635, 3.1624618
2: -4.8699226, -1.6805701, -4.8420539, -1.6859347, -3.1839881, 3.1614838
3: -1.7423849, 1.6700652, -1.7269850, 1.6645019, -3.4068868, 3.3970501
4: -14.0116749, -10.0041485, -14.0081949, -10.0282602, -3.7827702, 3.7998738
5: -8.5616531, -5.0880113, -8.5575190, -5.0969110, -2.6380296, 2.6432440
6: -12.7881174, -8.5354910, -12.7730389, -8.5379333, -3.8007431, 3.7867799
7: -9.2025137, -5.6786847, -9.1983738, -5.7004948, -3.3187685, 3.3374519
8: 9.6287947, 12.5982733, 9.6376295, 12.5816565, -2.9528618, 2.9606438
9: -7.9776421, -3.6958699, -7.9733357, -3.6992114, -3.4862852, 3.4858069

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688058, upper bound: 1.9621186
time: 4.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688875, upper bound: 1.9688875
time: 5.04 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.71 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9632779
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9688918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 8, lower bound: -1.9688058, upper bound: 1.9621186
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 24.71
Output dim: 8, lower bound: -1.9688875, upper bound: 1.9688875

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -11.0289745, -6.8591261, -11.0289745, -6.8591261, -3.6381569, 3.6381569
1: -9.9610510, -6.7885365, -9.9610510, -6.7885365, -3.1269269, 3.1269274
2: -4.8398800, -1.7017744, -4.8398800, -1.7017744, -3.1381056, 3.1381056
3: -1.7212706, 1.6580515, -1.7212706, 1.6580515, -3.3793221, 3.3793221
4: -13.9953804, -10.0299664, -13.9953804, -10.0299664, -3.7591991, 3.7591991
5: -8.5511646, -5.0987401, -8.5511646, -5.0987401, -2.6263437, 2.6263437
6: -12.7698078, -8.5451317, -12.7698078, -8.5451317, -3.7753906, 3.7753901
7: -9.1839590, -5.7023129, -9.1839590, -5.7023129, -3.2989664, 3.2989669
8: 9.6516819, 12.5805035, 9.6516819, 12.5805035, -2.9288216, 2.9288216
9: -7.9719172, -3.7013087, -7.9719172, -3.7013087, -3.4772053, 3.4772038

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9565051, upper bound: 1.9631921
time: 4.48 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632738, upper bound: 1.9632736
time: 5.01 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -11.0289745, -6.8591261, -11.0377007, -6.8517303, -3.6460805, 3.6466103
1: -9.9610510, -6.7885365, -9.9825935, -6.7680578, -3.1492934, 3.1491051
2: -4.8398800, -1.7017744, -4.8699226, -1.6805701, -3.1593099, 3.1681483
3: -1.7212706, 1.6580515, -1.7423849, 1.6700652, -3.3913357, 3.4004364
4: -13.9953804, -10.0299664, -14.0116749, -10.0041485, -3.7837296, 3.7749958
5: -8.5511646, -5.0987401, -8.5616531, -5.0880113, -2.6369119, 2.6367638
6: -12.7698078, -8.5451317, -12.7881174, -8.5354910, -3.7847757, 3.7935834
7: -9.1839590, -5.7023129, -9.2025137, -5.6786847, -3.3220911, 3.3169765
8: 9.6516819, 12.5805035, 9.6287947, 12.5982733, -2.9465914, 2.9517088
9: -7.9719172, -3.7013087, -7.9776421, -3.6958699, -3.4823050, 3.4830227

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9565051, upper bound: 1.9688060
time: 4.91 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632738, upper bound: 1.9688875
time: 4.92 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.0352716, -6.8630238, -10.9943504, -6.8959675, -3.6116319, 3.6123981
1: -9.9807749, -6.7700930, -9.9501467, -6.7939305, -3.1382694, 3.1342397
2: -4.8672051, -1.6823357, -4.8281307, -1.6998518, -3.1673534, 3.1457949
3: -1.7321606, 1.6670632, -1.6883125, 1.6242027, -3.3563633, 3.3553758
4: -14.0085411, -10.0139771, -13.9695158, -10.0641232, -3.7439222, 3.7550640
5: -8.5604410, -5.0933557, -8.5379581, -5.1176305, -2.6155868, 2.6177306
6: -12.7866640, -8.5380478, -12.7622223, -8.5495796, -3.7835999, 3.7683535
7: -9.1982985, -5.6801124, -9.1791868, -5.7054548, -3.3006973, 3.3163538
8: 9.6327028, 12.5969467, 9.6553745, 12.5647345, -2.9320316, 2.9415722
9: -7.9743605, -3.6991205, -7.9507093, -3.7133212, -3.4681373, 3.4597836

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9661652, upper bound: 1.9557067
time: 4.96 seconds

## Relational analysis of IS_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627972, upper bound: 1.9612496
time: 6.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687943, upper bound: 1.9621076
time: 5.05 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.0376987, -6.8517361, -11.0311966, -6.8543038, -3.6257124, 3.6575956
1: -9.9825916, -6.7680616, -9.9745512, -6.7861023, -3.1488352, 3.1584346
2: -4.8699217, -1.6805730, -4.8420467, -1.6859399, -3.1831856, 3.1614738
3: -1.7423759, 1.6700627, -1.7269561, 1.6644912, -3.4068670, 3.3970189
4: -14.0116749, -10.0041552, -14.0081854, -10.0282831, -3.7750816, 3.7927556
5: -8.5616503, -5.0880156, -8.5575132, -5.0969276, -2.6218243, 2.6432338
6: -12.7881193, -8.5354939, -12.7730370, -8.5379467, -3.8101759, 3.7781353
7: -9.2025127, -5.6786852, -9.1983604, -5.7004995, -3.3066282, 3.3523493
8: 9.6287975, 12.5982704, 9.6376429, 12.5816536, -2.9528561, 2.9606276
9: -7.9776421, -3.6958714, -7.9733262, -3.6992235, -3.4858303, 3.4897389

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9590448, upper bound: 1.9683023
time: 6.36 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9683020, upper bound: 1.9683025
time: 5.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.82 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9565051, upper bound: 1.9631921
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9632738, upper bound: 1.9632736
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9565051, upper bound: 1.9688060
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9632738, upper bound: 1.9688875
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9627972, upper bound: 1.9612496
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9687943, upper bound: 1.9621076
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9590448, upper bound: 1.9683023
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.82
Output dim: 8, lower bound: -1.9683020, upper bound: 1.9683025

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.9921265, -6.9008217, -11.0265465, -6.8704195, -3.5929484, 3.5948143
1: -9.9366169, -6.7963743, -9.9592409, -6.7905712, -3.0987749, 3.1144438
2: -4.8259635, -1.7156953, -4.8371596, -1.7035393, -3.1224241, 3.1214643
3: -1.6825666, 1.6177291, -1.7110248, 1.6550372, -3.3376038, 3.3287539
4: -13.9566727, -10.0658340, -13.9922333, -10.0397949, -3.7144203, 3.7203679
5: -8.5316105, -5.1194510, -8.5499535, -5.1040878, -2.6008210, 2.6039124
6: -12.7589989, -8.5567846, -12.7683582, -8.5476904, -3.7569714, 3.7582669
7: -9.1647539, -5.7072721, -9.1797352, -5.7037401, -3.2779140, 3.2809148
8: 9.6694355, 12.5635843, 9.6555929, 12.5791807, -2.9097452, 2.9079914
9: -7.9492970, -3.7154231, -7.9686475, -3.7045641, -3.4511881, 3.4590669

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9500963, upper bound: 1.9605515
time: 6.40 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9504837, upper bound: 1.9623215
time: 4.52 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9564937, upper bound: 1.9631810
time: 5.69 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.0289698, -6.8591528, -11.0289745, -6.8591318, -3.6381435, 3.6088963
1: -9.9610462, -6.7885456, -9.9610491, -6.7885385, -3.1269174, 3.1249967
2: -4.8398705, -1.7017800, -4.8398762, -1.7017765, -3.1380939, 3.1380963
3: -1.7212343, 1.6580404, -1.7212610, 1.6580493, -3.3792837, 3.3793015
4: -13.9953661, -10.0299864, -13.9953756, -10.0299683, -3.7591858, 3.7515101
5: -8.5511618, -5.0987597, -8.5511637, -5.0987434, -2.6263347, 2.6101370
6: -12.7698021, -8.5451374, -12.7698021, -8.5451317, -3.7667408, 3.7848392
7: -9.1839428, -5.7023191, -9.1839533, -5.7023168, -3.3138657, 3.2867975
8: 9.6516924, 12.5805016, 9.6516829, 12.5805025, -2.9288101, 2.9288187
9: -7.9719062, -3.7013190, -7.9719143, -3.7013113, -3.4811354, 3.4767475

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9534312
time: 5.39 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9626875
time: 4.87 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.9921265, -6.9008217, -11.0352716, -6.8630238, -3.6008763, 3.6032662
1: -9.9366169, -6.7963743, -9.9807749, -6.7700930, -3.1211414, 3.1366096
2: -4.8259635, -1.7156953, -4.8672051, -1.6823357, -3.1436276, 3.1510971
3: -1.6825666, 1.6177291, -1.7321606, 1.6670632, -3.3496299, 3.3498898
4: -13.9566727, -10.0658340, -14.0085411, -10.0139771, -3.7389498, 3.7361641
5: -8.5316105, -5.1194510, -8.5604410, -5.0933557, -2.6113939, 2.6143343
6: -12.7589989, -8.5567846, -12.7866640, -8.5380478, -3.7663574, 3.7764521
7: -9.1647539, -5.7072721, -9.1982985, -5.6801124, -3.3010349, 3.2989249
8: 9.6694355, 12.5635843, 9.6327028, 12.5969467, -2.9275112, 2.9308815
9: -7.9492970, -3.7154231, -7.9743605, -3.6991205, -3.4562893, 3.4648795

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9500963, upper bound: 1.9661653
time: 5.72 seconds

## Relational analysis of IS_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9556359, upper bound: 1.9627968
time: 5.07 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9564940, upper bound: 1.9687941
time: 4.86 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.0289698, -6.8591528, -11.0376987, -6.8517361, -3.6460662, 3.6173487
1: -9.9610462, -6.7885456, -9.9825916, -6.7680616, -3.1452441, 3.1471744
2: -4.8398705, -1.7017800, -4.8699217, -1.6805730, -3.1592975, 3.1649187
3: -1.7212343, 1.6580404, -1.7423759, 1.6700627, -3.3912969, 3.4004164
4: -13.9953661, -10.0299864, -14.0116749, -10.0041552, -3.7766099, 3.7673068
5: -8.5511618, -5.0987597, -8.5616503, -5.0880156, -2.6369019, 2.6205587
6: -12.7698021, -8.5451374, -12.7881193, -8.5354939, -3.7761273, 3.8030138
7: -9.1839428, -5.7023191, -9.2025127, -5.6786852, -3.3369894, 3.3048353
8: 9.6516924, 12.5805016, 9.6287975, 12.5982704, -2.9465780, 2.9517040
9: -7.9719062, -3.7013190, -7.9776421, -3.6958714, -3.4862361, 3.4825673

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9590445
time: 4.71 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9683014
time: 4.84 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.0009174, -6.8784881, -10.9861326, -6.8994675, -3.5639362, 3.5816813
1: -9.9720984, -6.7895575, -9.9481249, -6.7985768, -3.1235046, 3.1115055
2: -4.8623590, -1.6980295, -4.8269978, -1.7036217, -3.1516323, 3.1282101
3: -1.7176828, 1.6615769, -1.6848416, 1.6228718, -3.3405547, 3.3464184
4: -14.0009222, -10.0292492, -13.9677334, -10.0678034, -3.7276144, 3.7341967
5: -8.5497885, -5.0965672, -8.5354033, -5.1183863, -2.6026130, 2.6110859
6: -12.7727537, -8.5530739, -12.7586679, -8.5532207, -3.7647834, 3.7492223
7: -9.1901245, -5.7043419, -9.1773243, -5.7112317, -3.2839489, 3.2873964
8: 9.6523981, 12.5937834, 9.6600866, 12.5639935, -2.9115953, 2.9336967
9: -7.9615707, -3.7049127, -7.9476428, -3.7146833, -3.4526844, 3.4494629

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9601573, upper bound: 1.9548350
time: 4.71 seconds

## Relational analysis of IS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627970, upper bound: 1.9556359
time: 4.73 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627974, upper bound: 1.9556359
time: 5.11 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.0396605, -6.8301415, -10.9943314, -6.8959713, -3.6074662, 3.6545830
1: -10.0099134, -6.7672253, -9.9501429, -6.7939444, -3.1673861, 3.1356983
2: -4.8829870, -1.6793053, -4.8281288, -1.6998630, -3.1720371, 3.1488235
3: -1.7492623, 1.6770486, -1.6883016, 1.6241989, -3.3734612, 3.3653502
4: -14.0216331, -10.0042543, -13.9695129, -10.0641308, -3.7526331, 3.7631788
5: -8.5647564, -5.0829153, -8.5379505, -5.1176310, -2.6197090, 2.6275277
6: -12.7993908, -8.5303516, -12.7622156, -8.5495892, -3.7962914, 3.7751765
7: -9.2270241, -5.6739125, -9.1791830, -5.7054691, -3.3269253, 3.3211565
8: 9.6165209, 12.6077833, 9.6553860, 12.5647316, -2.9482107, 2.9523973
9: -7.9796853, -3.6856451, -7.9507008, -3.7133241, -3.4740934, 3.4744210

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9661538, upper bound: 1.9556958
time: 5.78 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687942, upper bound: 1.9564939
time: 5.28 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687945, upper bound: 1.9564940
time: 5.06 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0366764, -6.8560343, -11.0310583, -6.8548918, -3.6235523, 3.6508069
1: -9.9810429, -6.7746434, -9.9743423, -6.7869987, -3.1460247, 3.1506121
2: -4.8641005, -1.6817156, -4.8412499, -1.6860961, -3.1770463, 3.1595345
3: -1.7329242, 1.6658404, -1.7256682, 1.6639158, -3.3968401, 3.3915086
4: -14.0071068, -10.0171700, -14.0075645, -10.0300550, -3.7702990, 3.7788548
5: -8.5604172, -5.0982399, -8.5573463, -5.0983219, -2.6193948, 2.6328776
6: -12.7870407, -8.5389023, -12.7728910, -8.5384064, -3.8048296, 3.7756758
7: -9.1990767, -5.6800327, -9.1978922, -5.7006831, -3.3035250, 3.3438530
8: 9.6362104, 12.5968590, 9.6386509, 12.5814619, -2.9452515, 2.9582081
9: -7.9752455, -3.6967118, -7.9730010, -3.6993353, -3.4831347, 3.4861369

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9530283, upper bound: 1.9674276
time: 5.60 seconds

## Relational analysis of IS_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9590334, upper bound: 1.9682913
time: 5.97 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.0450478, -6.8505163, -11.0311890, -6.8543081, -3.6328125, 3.6553941
1: -10.0044031, -6.7672443, -9.9745502, -6.7861290, -3.1706676, 3.1583664
2: -4.8737469, -1.6706356, -4.8420401, -1.6859694, -3.1861372, 3.1714044
3: -1.7450938, 1.6873806, -1.7269168, 1.6644852, -3.4095790, 3.4142973
4: -14.0310287, -10.0029726, -14.0081797, -10.0283003, -3.7945852, 3.7927399
5: -8.5787506, -5.0860109, -8.5574923, -5.0969400, -2.6392846, 2.6437364
6: -12.7882452, -8.5322485, -12.7724419, -8.5379467, -3.8077040, 3.7888565
7: -9.2089157, -5.6813641, -9.1983557, -5.7035503, -3.3179741, 3.3520970
8: 9.6265373, 12.6106968, 9.6377077, 12.5816507, -2.9551134, 2.9729891
9: -7.9760723, -3.6967654, -7.9720497, -3.6992342, -3.4961615, 3.4845648

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9622888, upper bound: 1.9674280
time: 4.51 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9682906, upper bound: 1.9682915
time: 5.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.73 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9504837, upper bound: 1.9623215
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9564937, upper bound: 1.9631810
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9534312
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9626875
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9556359, upper bound: 1.9627968
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9564940, upper bound: 1.9687941
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9590445
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9626881, upper bound: 1.9683014
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9627970, upper bound: 1.9556359
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9627974, upper bound: 1.9556359
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9687942, upper bound: 1.9564939
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9687945, upper bound: 1.9564940
IS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9530283, upper bound: 1.9674276
IS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9590334, upper bound: 1.9682913
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9622888, upper bound: 1.9674280
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.73
Output dim: 8, lower bound: -1.9682906, upper bound: 1.9682915

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -10.9577341, -6.9162884, -11.0183392, -6.8739223, -3.5451899, 3.5642524
1: -9.9279137, -6.8158302, -9.9572277, -6.7952185, -3.0840206, 3.0917168
2: -4.8211370, -1.7314006, -4.8360238, -1.7073185, -3.1098514, 3.1046233
3: -1.6681552, 1.6121948, -1.7075765, 1.6537141, -3.3218694, 3.3197713
4: -13.9490433, -10.0811577, -13.9904423, -10.0434704, -3.6982012, 3.6994762
5: -8.5209389, -5.1226549, -8.5473967, -5.1048484, -2.5878854, 2.5972681
6: -12.7452774, -8.5718527, -12.7647896, -8.5513239, -3.7382579, 3.7390389
7: -9.1566687, -5.7314806, -9.1778584, -5.7095232, -3.2612658, 3.2519588
8: 9.6891937, 12.5604191, 9.6603088, 12.5784407, -2.8892469, 2.9001102
9: -7.9365120, -3.7212515, -7.9655838, -3.7059164, -3.4357791, 3.4487247

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9439738, upper bound: 1.9596817
time: 5.81 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9504837, upper bound: 1.9571834
time: 4.61 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9504837, upper bound: 1.9623215
time: 4.71 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -10.9965000, -6.8679819, -11.0265265, -6.8704247, -3.5887432, 3.6403670
1: -9.9657488, -6.7935257, -9.9592361, -6.7905850, -3.1276231, 3.1158862
2: -4.8416882, -1.7126814, -4.8371563, -1.7035499, -3.1362119, 3.1244750
3: -1.6996477, 1.6278374, -1.7110152, 1.6550348, -3.3546824, 3.3388526
4: -13.9698286, -10.0561867, -13.9922295, -10.0398026, -3.7230873, 3.7284765
5: -8.5359230, -5.1089954, -8.5499439, -5.1040874, -2.6049781, 2.6137137
6: -12.7717495, -8.5491095, -12.7683468, -8.5477009, -3.7696791, 3.7650585
7: -9.1933279, -5.7010574, -9.1797333, -5.7037544, -3.3039675, 3.2857332
8: 9.6532726, 12.5744305, 9.6556034, 12.5791788, -2.9259062, 2.9188271
9: -7.9546399, -3.7019691, -7.9686408, -3.7045648, -3.4571776, 3.4736838

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9500851, upper bound: 1.9605406
time: 5.51 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9497372, upper bound: 1.9622677
time: 5.15 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9564890, upper bound: 1.9631769
time: 4.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0288286, -6.8597407, -11.0279560, -6.8634295, -3.6313529, 3.6067371
1: -9.9608364, -6.7894402, -9.9595127, -6.7951207, -3.1191120, 3.1221819
2: -4.8390751, -1.7019358, -4.8340540, -1.7029206, -3.1361547, 3.1321182
3: -1.7199426, 1.6574643, -1.7117829, 1.6538110, -3.3737535, 3.3692472
4: -13.9947433, -10.0317593, -13.9907866, -10.0429888, -3.7453260, 3.7467537
5: -8.5509949, -5.1001549, -8.5499306, -5.1089745, -2.6159711, 2.6077099
6: -12.7696533, -8.5456028, -12.7687206, -8.5485525, -3.7642789, 3.7794933
7: -9.1834774, -5.7025037, -9.1805105, -5.7036662, -3.3053713, 3.2837243
8: 9.6527014, 12.5803108, 9.6590948, 12.5790949, -2.9263935, 2.9212160
9: -7.9715829, -3.7014346, -7.9695344, -3.7021472, -3.4775290, 3.4740615

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9566757, upper bound: 1.9525587
time: 4.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626768, upper bound: 1.9534202
time: 4.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.0289640, -6.8591576, -11.0363274, -6.8579111, -3.6359377, 3.6159973
1: -9.9610424, -6.7885709, -9.9828892, -6.7877145, -3.1263366, 3.1467080
2: -4.8398638, -1.7018101, -4.8436990, -1.6918375, -3.1480262, 3.1418889
3: -1.7211928, 1.6580369, -1.7239819, 1.6753950, -3.3965878, 3.3820188
4: -13.9953613, -10.0300055, -14.0147438, -10.0287895, -3.7591524, 3.7710161
5: -8.5511379, -5.0987711, -8.5682621, -5.0967321, -2.6268411, 2.6276028
6: -12.7692070, -8.5451441, -12.7699099, -8.5418663, -3.7774734, 3.7823529
7: -9.1839390, -5.7053680, -9.1903925, -5.7049923, -3.3136148, 3.2981524
8: 9.6517582, 12.5804968, 9.6494102, 12.5929356, -2.9411774, 2.9310865
9: -7.9706311, -3.7013340, -7.9703579, -3.7021871, -3.4759626, 3.4870863

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9566757, upper bound: 1.9618143
time: 4.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626768, upper bound: 1.9626765
time: 4.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.9839087, -6.9043241, -11.0009174, -6.8784881, -3.5701704, 3.5554762
1: -9.9345970, -6.8010211, -9.9720984, -6.7895575, -3.0984030, 3.1218452
2: -4.8248320, -1.7194712, -4.8623590, -1.6980295, -3.1248875, 3.1333742
3: -1.6791186, 1.6163979, -1.7176828, 1.6615769, -3.3406954, 3.3340807
4: -13.9548883, -10.0695124, -14.0009222, -10.0292492, -3.7180824, 3.7199059
5: -8.5290575, -5.1202106, -8.5497885, -5.0965672, -2.6047497, 2.6014063
6: -12.7554522, -8.5604267, -12.7727537, -8.5530739, -3.7472334, 3.7576361
7: -9.1628895, -5.7130504, -9.1901245, -5.7043419, -3.2720785, 3.2821660
8: 9.6741562, 12.5628433, 9.6523981, 12.5937834, -2.9196272, 2.9104452
9: -7.9462318, -3.7167845, -7.9615707, -3.7049127, -3.4459972, 3.4494267

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9492273, upper bound: 1.9601569
time: 5.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9504837, upper bound: 1.9627970
time: 5.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9504837, upper bound: 1.9627967
time: 5.55 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.9921064, -6.9008269, -11.0396605, -6.8301415, -3.6463165, 3.5990996
1: -9.9366131, -6.7963881, -10.0099134, -6.7672253, -3.1226006, 3.1656358
2: -4.8259621, -1.7157066, -4.8829870, -1.6793053, -3.1466568, 3.1537709
3: -1.6825564, 1.6177263, -1.7492623, 1.6770486, -3.3596048, 3.3669887
4: -13.9566708, -10.0658407, -14.0216331, -10.0042543, -3.7470636, 3.7448378
5: -8.5316038, -5.1194544, -8.5647564, -5.0829153, -2.6211910, 2.6184731
6: -12.7589903, -8.5567942, -12.7993908, -8.5303516, -3.7731814, 3.7891459
7: -9.1647472, -5.7072840, -9.2270241, -5.6739125, -3.3058386, 3.3250933
8: 9.6694489, 12.5635834, 9.6165209, 12.6077833, -2.9383345, 2.9470625
9: -7.9492912, -3.7154269, -7.9796853, -3.6856451, -3.4709272, 3.4708343

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9500854, upper bound: 1.9661534
time: 4.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9555843, upper bound: 1.9620583
time: 4.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9564896, upper bound: 1.9687895
time: 5.14 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.0288286, -6.8597407, -11.0366764, -6.8560343, -3.6392813, 3.6151891
1: -9.9608364, -6.7894402, -9.9810429, -6.7746434, -3.1374240, 3.1443663
2: -4.8390751, -1.7019358, -4.8641005, -1.6817156, -3.1573596, 3.1587794
3: -1.7199426, 1.6574643, -1.7329242, 1.6658404, -3.3857830, 3.3903885
4: -13.9947433, -10.0317593, -14.0071068, -10.0171700, -3.7627106, 3.7625523
5: -8.5509949, -5.1001549, -8.5604172, -5.0982399, -2.6265450, 2.6181335
6: -12.7696533, -8.5456028, -12.7870407, -8.5389023, -3.7736673, 3.7976675
7: -9.1834774, -5.7025037, -9.1990767, -5.6800327, -3.3284950, 3.3017607
8: 9.6527014, 12.5803108, 9.6362104, 12.5968590, -2.9441576, 2.9441004
9: -7.9715829, -3.7014346, -7.9752455, -3.6967118, -3.4826336, 3.4798741

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9618149, upper bound: 1.9530281
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626771, upper bound: 1.9590332
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.0289640, -6.8591576, -11.0450478, -6.8505163, -3.6438637, 3.6244488
1: -9.9610424, -6.7885709, -10.0044031, -6.7672443, -3.1451769, 3.1690078
2: -4.8398638, -1.7018101, -4.8737469, -1.6706356, -3.1692281, 3.1678705
3: -1.7211928, 1.6580369, -1.7450938, 1.6873806, -3.4085734, 3.4031308
4: -13.9953613, -10.0300055, -14.0310287, -10.0029726, -3.7765942, 3.7868075
5: -8.5511379, -5.0987711, -8.5787506, -5.0860109, -2.6374040, 2.6380243
6: -12.7692070, -8.5451441, -12.7882452, -8.5322485, -3.7868524, 3.8005428
7: -9.1839390, -5.7053680, -9.2089157, -5.6813641, -3.3367376, 3.3161702
8: 9.6517582, 12.5804968, 9.6265373, 12.6106968, -2.9589386, 2.9539595
9: -7.9706311, -3.7013340, -7.9760723, -3.6967654, -3.4810505, 3.4929004

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9618149, upper bound: 1.9622883
time: 4.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626771, upper bound: 1.9682900
time: 4.61 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.0009174, -6.8784881, -10.9839087, -6.9043241, -3.5554762, 3.5701709
1: -9.9720984, -6.7895575, -9.9345970, -6.8010211, -3.1218457, 3.0984035
2: -4.8623590, -1.6980295, -4.8248320, -1.7194712, -3.1333742, 3.1248875
3: -1.7176828, 1.6615769, -1.6791186, 1.6163979, -3.3340807, 3.3406954
4: -14.0009222, -10.0292492, -13.9548883, -10.0695124, -3.7199059, 3.7180820
5: -8.5497885, -5.0965672, -8.5290575, -5.1202106, -2.6014066, 2.6047497
6: -12.7727537, -8.5530739, -12.7554522, -8.5604267, -3.7576356, 3.7472324
7: -9.1901245, -5.7043419, -9.1628895, -5.7130504, -3.2821655, 3.2720785
8: 9.6523981, 12.5937834, 9.6741562, 12.5628433, -2.9104452, 2.9196272
9: -7.9615707, -3.7049127, -7.9462318, -3.7167845, -3.4494267, 3.4459972

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9601571, upper bound: 1.9492270
time: 5.39 seconds

## Relational analysis of IS_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.944035530090332
rel_dist={8: [-1.969206455838302, 1.969206699419745]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5845
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205467, upper bound: 1.6237266
time: 5.98 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237271, upper bound: 1.6237266
time: 5.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.39 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.39
Output dim: 8, lower bound: -1.6205467, upper bound: 1.6237266
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.39
Output dim: 8, lower bound: -1.6237271, upper bound: 1.6237266

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -11.0289745, -6.8591261, -11.0301819, -6.8565211, -3.2446551, 3.2425556
1: -9.9610510, -6.7885365, -9.9683094, -6.7872171, -2.8730154, 2.8784776
2: -4.8398800, -1.7017744, -4.8410511, -1.6932533, -2.9190226, 2.9104753
3: -1.7212706, 1.6580515, -1.7243526, 1.6615148, -3.3827853, 3.3824041
4: -13.9953804, -10.0299664, -14.0022669, -10.0290432, -3.3991632, 3.4055576
5: -8.5511646, -5.0987401, -8.5545769, -5.0977526, -2.2768798, 2.2794354
6: -12.7698078, -8.5451317, -12.7715511, -8.5412664, -3.4133101, 3.4113541
7: -9.1839590, -5.7023129, -9.1917067, -5.7013302, -2.9506893, 2.9574099
8: 9.6516819, 12.5805035, 9.6441193, 12.5811272, -2.7491379, 2.7574329
9: -7.9719172, -3.7013087, -7.9726877, -3.7001834, -3.1173840, 3.1163969

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6191775
time: 6.72 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6237239
time: 6.23 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.0377007, -6.8517303, -11.0312042, -6.8542762, -3.2569695, 3.2577448
1: -9.9825935, -6.7680578, -9.9745522, -6.7860951, -2.8938298, 2.9058166
2: -4.8699226, -1.6805701, -4.8420525, -1.6859405, -2.9506664, 2.9279923
3: -1.7423849, 1.6700652, -1.7269826, 1.6644995, -3.4068844, 3.3970478
4: -14.0116749, -10.0041485, -14.0081940, -10.0282612, -3.4159155, 3.4375348
5: -8.5616531, -5.0880113, -8.5575161, -5.0969095, -2.2864976, 2.2929175
6: -12.7881174, -8.5354910, -12.7730389, -8.5379343, -3.4348135, 3.4192796
7: -9.2025137, -5.6786847, -9.1983700, -5.7004952, -2.9652100, 2.9876156
8: 9.6287947, 12.5982733, 9.6376390, 12.5816574, -2.7732224, 2.7833445
9: -7.9776421, -3.6958699, -7.9733372, -3.6992140, -3.1247063, 3.1237741

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237240, upper bound: 1.6191780
time: 5.58 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237240, upper bound: 1.6237237
time: 5.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.68
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6191775
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.68
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6237239
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.68
Output dim: 8, lower bound: -1.6237240, upper bound: 1.6191780
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.68
Output dim: 8, lower bound: -1.6237240, upper bound: 1.6237237

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -11.0246716, -6.8790312, -10.9933310, -6.8982124, -3.1999345, 3.1888566
1: -9.9578495, -6.7921200, -9.9438877, -6.7950544, -2.8589611, 2.8482261
2: -4.8350897, -1.7048960, -4.8271332, -1.7071732, -2.8998423, 2.8920441
3: -1.7032228, 1.6527251, -1.6856642, 1.6212060, -3.3244288, 3.3383894
4: -13.9898119, -10.0472879, -13.9635763, -10.0649080, -3.3588543, 3.3529787
5: -8.5490189, -5.1081600, -8.5350151, -5.1184664, -2.2535877, 2.2497671
6: -12.7672482, -8.5496292, -12.7607403, -8.5529127, -3.3948927, 3.3915982
7: -9.1765356, -5.7048197, -9.1725130, -5.7062874, -2.9314556, 2.9347563
8: 9.6585751, 12.5781612, 9.6618681, 12.5642071, -2.7239919, 2.7354605
9: -7.9661417, -3.7070405, -7.9500647, -3.7142944, -3.0968361, 3.0871906

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165416, upper bound: 1.6178916
time: 5.99 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205371, upper bound: 1.6191717
time: 5.67 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -11.0289736, -6.8591390, -11.0301743, -6.8565493, -3.2102242, 3.2425327
1: -9.9610481, -6.7885413, -9.9683046, -6.7872233, -2.8707452, 2.8784661
2: -4.8398752, -1.7017778, -4.8410430, -1.6932606, -2.9190130, 2.9061451
3: -1.7212529, 1.6580458, -1.7243166, 1.6615043, -3.3827572, 3.3823624
4: -13.9953737, -10.0299740, -14.0022573, -10.0290651, -3.3901205, 3.4055347
5: -8.5511618, -5.0987501, -8.5545721, -5.0977726, -2.2578311, 2.2794209
6: -12.7698050, -8.5451355, -12.7715464, -8.5412750, -3.4201722, 3.4003644
7: -9.1839514, -5.7023153, -9.1916904, -5.7013345, -2.9385157, 2.9682684
8: 9.6516867, 12.5805016, 9.6441288, 12.5811253, -2.7491255, 2.7493405
9: -7.9719110, -3.7013149, -7.9726772, -3.7001920, -3.1164026, 3.1197476

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6140654, upper bound: 1.6231138
time: 5.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6199361, upper bound: 1.6231136
time: 6.08 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.0333939, -6.8716316, -10.9943495, -6.8959680, -3.2122450, 3.2040424
1: -9.9793749, -6.7716408, -9.9501429, -6.7939305, -2.8797579, 2.8742440
2: -4.8651381, -1.6836935, -4.8281298, -1.6998559, -2.9306898, 2.9095564
3: -1.7243605, 1.6647615, -1.6883121, 1.6242012, -3.3485618, 3.3530736
4: -14.0061264, -10.0214682, -13.9695110, -10.0641232, -3.3755751, 3.3849411
5: -8.5595074, -5.0974288, -8.5379543, -5.1176281, -2.2631922, 2.2632575
6: -12.7855482, -8.5399857, -12.7622185, -8.5495815, -3.4163804, 3.3995190
7: -9.1950989, -5.6811910, -9.1791830, -5.7054543, -2.9459448, 2.9649386
8: 9.6356850, 12.5959263, 9.6553822, 12.5647354, -2.7480779, 2.7613697
9: -7.9718409, -3.7015998, -7.9507074, -3.7133222, -3.1041298, 3.0945663

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6197180, upper bound: 1.6178914
time: 6.21 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237174, upper bound: 1.6191717
time: 4.99 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.0376968, -6.8517437, -11.0311947, -6.8543057, -3.2225366, 3.2577214
1: -9.9825907, -6.7680635, -9.9745474, -6.7861028, -2.8915586, 2.8994653
2: -4.8699188, -1.6805737, -4.8420420, -1.6859448, -2.9467635, 2.9236622
3: -1.7423699, 1.6700600, -1.7269540, 1.6644881, -3.4068580, 3.3970141
4: -14.0116711, -10.0041552, -14.0081816, -10.0282841, -3.4068718, 3.4251800
5: -8.5616503, -5.0880208, -8.5575123, -5.0969300, -2.2674479, 2.2908061
6: -12.7881165, -8.5354948, -12.7730350, -8.5379448, -3.4416614, 3.4082937
7: -9.2025070, -5.6786861, -9.1983538, -5.7004995, -2.9530678, 2.9984722
8: 9.6287994, 12.5982685, 9.6376486, 12.5816536, -2.7732081, 2.7752516
9: -7.9776378, -3.6958747, -7.9733272, -3.6992221, -3.1237278, 3.1271262

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6172423, upper bound: 1.6231135
time: 6.09 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6231138, upper bound: 1.6231138
time: 6.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.02 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6165416, upper bound: 1.6178916
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6205371, upper bound: 1.6191717
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6140654, upper bound: 1.6231138
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6199361, upper bound: 1.6231136
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6197180, upper bound: 1.6178914
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6237174, upper bound: 1.6191717
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6172423, upper bound: 1.6231135
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 27.02
Output dim: 8, lower bound: -1.6231138, upper bound: 1.6231138

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.9903212, -6.8944950, -10.9765434, -6.9054890, -3.1472178, 3.1476226
1: -9.9491615, -6.8115773, -9.9397221, -6.8045454, -2.8391304, 2.8230643
2: -4.8302569, -1.7206150, -4.8248053, -1.7148658, -2.8789945, 2.8676801
3: -1.6888011, 1.6472222, -1.6786122, 1.6184922, -3.3072934, 3.3258343
4: -13.9821796, -10.0625982, -13.9599066, -10.0724182, -3.3380871, 3.3291306
5: -8.5383434, -5.1113729, -8.5298042, -5.1200218, -2.2397213, 2.2401104
6: -12.7534637, -8.5646791, -12.7536983, -8.5603247, -3.3722000, 3.3686328
7: -9.1684074, -5.7290363, -9.1686621, -5.7180920, -2.9081287, 2.9032965
8: 9.6783295, 12.5749989, 9.6715012, 12.5626822, -2.7005239, 2.7191939
9: -7.9533625, -3.7128494, -7.9438086, -3.7170947, -3.0796657, 3.0734425

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6136535, upper bound: 1.6174910
time: 5.02 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6178885
time: 5.71 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.0290585, -6.8461738, -10.9932957, -6.8982286, -3.1957455, 3.2281761
1: -9.9869938, -6.7892675, -9.9438791, -6.7950764, -2.8862729, 2.8480573
2: -4.8508468, -1.7018845, -4.8271294, -1.7071931, -2.9100299, 2.8939052
3: -1.7203324, 1.6627629, -1.6856484, 1.6212012, -3.3415337, 3.3484113
4: -14.0029135, -10.0375957, -13.9635706, -10.0649204, -3.3674974, 3.3590136
5: -8.5533199, -5.0977192, -8.5350018, -5.1184707, -2.2572384, 2.2595615
6: -12.7800045, -8.5419159, -12.7607288, -8.5529327, -3.4075966, 3.3981318
7: -9.2052450, -5.6986146, -9.1725073, -5.7063117, -2.9575977, 2.9369097
8: 9.6424341, 12.5889883, 9.6618872, 12.5642033, -2.7358813, 2.7450063
9: -7.9714746, -3.6935654, -7.9500499, -3.7143011, -3.1028023, 3.1017942

Time for backsubstitution: 14.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 4627

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6176574, upper bound: 1.6187748
time: 5.68 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6191682
time: 6.14 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.0279541, -6.8634367, -11.0297356, -6.8583984, -3.2061243, 3.2353735
1: -9.9595127, -6.7951250, -9.9676437, -6.7900496, -2.8657117, 2.8701344
2: -4.8340497, -1.7029221, -4.8385391, -1.6937510, -2.9125452, 2.9024873
3: -1.7117763, 1.6538064, -1.7202532, 1.6596907, -3.3714671, 3.3740597
4: -13.9907875, -10.0429945, -14.0002937, -10.0346575, -3.3814125, 3.3908181
5: -8.5499287, -5.1089807, -8.5540409, -5.1021643, -2.2524061, 2.2687547
6: -12.7687225, -8.5485554, -12.7710838, -8.5427380, -3.4141541, 3.3975625
7: -9.1805058, -5.7036676, -9.1902180, -5.7019119, -2.9348736, 2.9591331
8: 9.6590986, 12.5790939, 9.6473141, 12.5805206, -2.7411437, 2.7448940
9: -7.9695330, -3.7021508, -7.9716578, -3.7005515, -3.1127448, 3.1154871

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6127760, upper bound: 1.6191065
time: 5.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6140592, upper bound: 1.6231072
time: 5.73 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.0363216, -6.8579187, -11.0301657, -6.8565540, -3.2173204, 3.2380981
1: -9.9828873, -6.7877164, -9.9682989, -6.7872696, -2.8896914, 2.8756042
2: -4.8436975, -1.6918402, -4.8410325, -1.6933101, -2.9205208, 2.9162846
3: -1.7239745, 1.6753924, -1.7242460, 1.6614952, -3.3854697, 3.3996384
4: -14.0147381, -10.0287971, -14.0022478, -10.0290976, -3.4092593, 3.4041233
5: -8.5682621, -5.0967360, -8.5545311, -5.0977898, -2.2743442, 2.2771683
6: -12.7699089, -8.5419970, -12.7705421, -8.5412807, -3.4166384, 3.4100542
7: -9.1891356, -5.7049913, -9.1916866, -5.7046905, -2.9460535, 2.9678502
8: 9.6494112, 12.5929356, 9.6442423, 12.5811205, -2.7496262, 2.7620707
9: -7.9703550, -3.7021902, -7.9705262, -3.7002125, -3.1251125, 3.1128750

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6186475, upper bound: 1.6191064
time: 4.73 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6199299, upper bound: 1.6231070
time: 5.47 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.9990406, -6.8870945, -10.9775639, -6.9032450, -3.1595597, 3.1626320
1: -9.9706926, -6.7911081, -9.9459763, -6.8034267, -2.8599052, 2.8491392
2: -4.8602972, -1.6993825, -4.8258009, -1.7075441, -2.9080534, 2.8851876
3: -1.7098513, 1.6592717, -1.6812477, 1.6214855, -3.3313367, 3.3405194
4: -13.9985189, -10.0367765, -13.9658451, -10.0716305, -3.3547812, 3.3610883
5: -8.5488567, -5.1006393, -8.5327435, -5.1191835, -2.2492943, 2.2536080
6: -12.7716494, -8.5550213, -12.7551870, -8.5569906, -3.3936424, 3.3765683
7: -9.1869478, -5.7054205, -9.1753340, -5.7172623, -2.9225922, 2.9334602
8: 9.6553812, 12.5927649, 9.6650066, 12.5632105, -2.7245607, 2.7451081
9: -7.9590597, -3.7073860, -7.9444528, -3.7161212, -3.0869560, 3.0808139

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6193167, upper bound: 1.6150036
time: 5.57 seconds

## Relational analysis of IS_A2_B1_A1_A2

### Relational analysis result of IS_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6197149, upper bound: 1.6178882
time: 5.87 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.0377827, -6.8387585, -10.9943142, -6.8959761, -3.2080584, 3.2383904
1: -10.0085182, -6.7687736, -9.9501343, -6.7939548, -2.9072943, 2.8741598
2: -4.8809090, -1.6806630, -4.8281269, -1.6998756, -2.9333496, 2.9113832
3: -1.7414522, 1.6747602, -1.6882944, 1.6241963, -3.3656485, 3.3630548
4: -14.0192156, -10.0117798, -13.9695072, -10.0641394, -3.3842564, 3.3909650
5: -8.5638247, -5.0869913, -8.5379410, -5.1176305, -2.2668409, 2.2730510
6: -12.7982769, -8.5322876, -12.7622089, -8.5495987, -3.4290586, 3.4060450
7: -9.2237892, -5.6749926, -9.1791782, -5.7054796, -2.9721403, 2.9670820
8: 9.6194992, 12.6067638, 9.6554012, 12.5647297, -2.7599349, 2.7708039
9: -7.9771767, -3.6881316, -7.9506960, -3.7133257, -3.1101050, 3.1091633

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 4627

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6168300, upper bound: 1.6187748
time: 6.78 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237142, upper bound: 1.6191687
time: 4.69 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0366764, -6.8560429, -11.0307589, -6.8561535, -3.2184339, 3.2505608
1: -9.9810410, -6.7746449, -9.9738874, -6.7889295, -2.8865328, 2.8911130
2: -4.8640981, -1.6817188, -4.8395414, -1.6864349, -2.9403124, 2.9200020
3: -1.7329168, 1.6658385, -1.7228942, 1.6626778, -3.3955946, 3.3887327
4: -14.0071058, -10.0171776, -14.0062218, -10.0338755, -3.3981400, 3.4103980
5: -8.5604181, -5.0982442, -8.5569839, -5.1013203, -2.2620225, 2.2800915
6: -12.7870388, -8.5389061, -12.7725716, -8.5394077, -3.4356394, 3.4054914
7: -9.1990738, -5.6800332, -9.1968803, -5.7010789, -2.9493942, 2.9893341
8: 9.6362133, 12.5968580, 9.6408310, 12.5810499, -2.7652297, 2.7707984
9: -7.9752426, -3.6967132, -7.9723015, -3.6995802, -3.1200628, 3.1228662

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159518, upper bound: 1.6191065
time: 5.83 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6172360, upper bound: 1.6231071
time: 5.72 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.0450439, -6.8505216, -11.0311832, -6.8543148, -3.2296305, 3.2532878
1: -10.0044022, -6.7672462, -9.9745436, -6.7861476, -2.9106503, 2.8971059
2: -4.8737435, -1.6706375, -4.8420343, -1.6859951, -2.9483252, 2.9338017
3: -1.7450867, 1.6873760, -1.7268858, 1.6644796, -3.4095664, 3.4142618
4: -14.0310240, -10.0029793, -14.0081768, -10.0283136, -3.4222102, 3.4237785
5: -8.5787477, -5.0860176, -8.5574732, -5.0969458, -2.2834752, 2.2885263
6: -12.7882433, -8.5323792, -12.7720327, -8.5379534, -3.4381394, 3.4179721
7: -9.2076550, -5.6813641, -9.1983490, -5.7038565, -2.9605951, 2.9980550
8: 9.6265411, 12.6106977, 9.6377611, 12.5816498, -2.7737141, 2.7813113
9: -7.9760685, -3.6967690, -7.9711766, -3.6992402, -3.1324315, 3.1202502

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6191064, upper bound: 1.6218229
time: 12.07 seconds

## Relational analysis of IS_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6231072, upper bound: 1.6231073
time: 5.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 32.83 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6136535, upper bound: 1.6174910
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6178885
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6176574, upper bound: 1.6187748
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6191682
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6127760, upper bound: 1.6191065
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6140592, upper bound: 1.6231072
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6186475, upper bound: 1.6191064
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6199299, upper bound: 1.6231070
IS_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6193167, upper bound: 1.6150036
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6197149, upper bound: 1.6178882
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6168300, upper bound: 1.6187748
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6237142, upper bound: 1.6191687
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6159518, upper bound: 1.6191065
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6172360, upper bound: 1.6231071
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6191064, upper bound: 1.6218229
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 32.83
Output dim: 8, lower bound: -1.6231072, upper bound: 1.6231073

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.9789343, -6.8977456, -10.9525623, -6.9210949, -3.1180997, 3.1253853
1: -9.9432640, -6.8236957, -9.9161816, -6.8308635, -2.8061819, 2.7842712
2: -4.8234863, -1.7346236, -4.8042145, -1.7438583, -2.8447051, 2.8324547
3: -1.6674213, 1.6431839, -1.6310649, 1.5902452, -3.2576666, 3.2742488
4: -13.9668179, -10.0650101, -13.9299240, -10.0785971, -3.3106031, 3.2744665
5: -8.5292835, -5.1139784, -8.5104370, -5.1346540, -2.1986938, 2.2164278
6: -12.7404442, -8.5668278, -12.7271042, -8.5716686, -3.3463378, 3.3395486
7: -9.1578045, -5.7313910, -9.1455069, -5.7263532, -2.8852768, 2.8581529
8: 9.6848574, 12.5622759, 9.6901035, 12.5367498, -2.6574607, 2.6824493
9: -7.9477091, -3.7272546, -7.9191580, -3.7474599, -3.0399384, 3.0124640

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 6253

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6136534, upper bound: 1.6141868
time: 5.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6136534, upper bound: 1.6174910
time: 5.42 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.9903183, -6.8944960, -10.9765329, -6.9054937, -3.1490927, 3.1472902
1: -9.9491634, -6.8115830, -9.9397192, -6.8045640, -2.8386450, 2.8258138
2: -4.8302546, -1.7206218, -4.8247986, -1.7148856, -2.8789778, 2.8679481
3: -1.6887927, 1.6472206, -1.6785948, 1.6184862, -3.3072789, 3.3258154
4: -13.9821739, -10.0625992, -13.9598970, -10.0724201, -3.3215046, 3.3548937
5: -8.5383425, -5.1113744, -8.5297976, -5.1200228, -2.2397156, 2.2373667
6: -12.7534599, -8.5646791, -12.7536907, -8.5603256, -3.3721933, 3.3604035
7: -9.1684055, -5.7290359, -9.1686563, -5.7180977, -2.9008312, 2.9108925
8: 9.6783304, 12.5749960, 9.6715059, 12.5626745, -2.6951704, 2.7128453
9: -7.9533629, -3.7128510, -7.9438024, -3.7171080, -3.0776467, 3.0734334

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 6253

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6145810
time: 5.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6178886
time: 5.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0176706, -6.8494349, -10.9693127, -6.9138379, -3.1666489, 3.2061262
1: -9.9810991, -6.8013844, -9.9203415, -6.8213911, -2.8532753, 2.8092742
2: -4.8440742, -1.7158908, -4.8065376, -1.7361876, -2.8757272, 2.8586826
3: -1.6989403, 1.6587479, -1.6380939, 1.5929646, -3.2919049, 3.2968419
4: -13.9875479, -10.0400133, -13.9335785, -10.0710993, -3.3399982, 3.3043957
5: -8.5442638, -5.1003203, -8.5156460, -5.1331067, -2.2161417, 2.2358809
6: -12.7669773, -8.5440512, -12.7341576, -8.5642719, -3.3817501, 3.3690410
7: -9.1946163, -5.7009468, -9.1493464, -5.7145658, -2.9347401, 2.8917770
8: 9.6489210, 12.5762615, 9.6804953, 12.5382700, -2.6928730, 2.7082794
9: -7.9658270, -3.7079680, -7.9254041, -3.7446609, -3.0630875, 3.0408130

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6176576, upper bound: 1.6154697
time: 5.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6176576, upper bound: 1.6187747
time: 6.25 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.0290546, -6.8461752, -10.9932842, -6.8982277, -3.1976213, 3.2260842
1: -9.9869938, -6.7892752, -9.9438772, -6.7950954, -2.8826723, 2.8508062
2: -4.8508444, -1.7018907, -4.8271246, -1.7072127, -2.9100151, 2.8941736
3: -1.7203236, 1.6627603, -1.6856313, 1.6211982, -3.3415217, 3.3483915
4: -14.0029106, -10.0375977, -13.9635582, -10.0649242, -3.3509426, 3.3848014
5: -8.5533199, -5.0977182, -8.5349979, -5.1184711, -2.2572331, 2.2568192
6: -12.7800026, -8.5419178, -12.7607193, -8.5529337, -3.4075899, 3.3899002
7: -9.2052431, -5.6986146, -9.1725016, -5.7063165, -2.9503002, 2.9445591
8: 9.6424351, 12.5889854, 9.6618938, 12.5641966, -2.7305899, 2.7386639
9: -7.9714689, -3.6935666, -7.9500475, -3.7143137, -3.1007838, 3.1017861

Time for backsubstitution: 14.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6158606
time: 5.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6191682
time: 5.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -11.0111961, -6.8707056, -10.9953976, -6.8738637, -3.1649241, 3.1825981
1: -9.9553614, -6.8046141, -9.9589672, -6.8095064, -2.8405704, 2.8502860
2: -4.8317189, -1.7106340, -4.8336968, -1.7094654, -2.8881989, 2.8816538
3: -1.7047529, 1.6511059, -1.7058687, 1.6541914, -3.3589444, 3.3569746
4: -13.9871054, -10.0504923, -13.9926443, -10.0499001, -3.3575525, 3.3700414
5: -8.5447121, -5.1105413, -8.5433712, -5.1053824, -2.2427397, 2.2548738
6: -12.7616329, -8.5559559, -12.7572899, -8.5577726, -3.3912125, 3.3748455
7: -9.1766281, -5.7154803, -9.1820612, -5.7261357, -2.9033842, 2.9357638
8: 9.6687355, 12.5775700, 9.6670475, 12.5773582, -2.7248802, 2.7214379
9: -7.9632726, -3.7049408, -7.9588699, -3.7063570, -3.0989799, 3.0982976

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6123744, upper bound: 1.6162186
time: 5.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6127730, upper bound: 1.6191032
time: 4.93 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -11.0279179, -6.8634462, -11.0341320, -6.8255286, -3.2467489, 3.2312069
1: -9.9595051, -6.7951469, -9.9967823, -6.7871957, -2.8655396, 2.8901303
2: -4.8340468, -1.7029405, -4.8543234, -1.6907359, -2.9143996, 2.9127369
3: -1.7117589, 1.6538022, -1.7373748, 1.6697108, -3.3814697, 3.3911769
4: -13.9907789, -10.0430088, -14.0133972, -10.0249119, -3.3874502, 3.3977523
5: -8.5499163, -5.1089816, -8.5583363, -5.0917215, -2.2621937, 2.2723780
6: -12.7687111, -8.5485725, -12.7838573, -8.5350218, -3.4206648, 3.4102578
7: -9.1805010, -5.7036943, -9.2189922, -5.6957073, -2.9370279, 2.9853082
8: 9.6591177, 12.5790892, 9.6311750, 12.5913448, -2.7506895, 2.7567649
9: -7.9695206, -3.7021542, -7.9769869, -3.6870587, -3.1273742, 3.1214342

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 4627

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6136586, upper bound: 1.6202186
time: 5.02 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6140562, upper bound: 1.6231040
time: 5.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.0195675, -6.8651929, -10.9958229, -6.8720274, -3.1761193, 3.1853228
1: -9.9787331, -6.7972074, -9.9596252, -6.8067265, -2.8646164, 2.8557572
2: -4.8413553, -1.6995486, -4.8361883, -1.7090245, -2.8961782, 2.8954525
3: -1.7169313, 1.6727041, -1.7098730, 1.6560025, -3.3729339, 3.3825769
4: -14.0110540, -10.0362902, -13.9945879, -10.0443268, -3.3850460, 3.3833532
5: -8.5630465, -5.0982952, -8.5438633, -5.1010060, -2.2637210, 2.2632875
6: -12.7628155, -8.5493860, -12.7567482, -8.5563107, -3.3937130, 3.3874173
7: -9.1852436, -5.7168069, -9.1835136, -5.7289100, -2.9145269, 2.9444656
8: 9.6590471, 12.5914116, 9.6639767, 12.5779552, -2.7333679, 2.7386196
9: -7.9641013, -3.7049839, -7.9577498, -3.7060201, -3.1113262, 3.0956774

Time for backsubstitution: 14.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6182459, upper bound: 1.6162182
time: 5.56 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6186445, upper bound: 1.6191030
time: 5.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.0362892, -6.8579264, -11.0345592, -6.8236804, -3.2533293, 3.2339334
1: -9.9828796, -6.7877398, -9.9974403, -6.7844143, -2.8895969, 2.8961265
2: -4.8436918, -1.6918583, -4.8568201, -1.6902940, -2.9223742, 2.9255037
3: -1.7239575, 1.6753888, -1.7413855, 1.6715007, -3.3954582, 3.4167743
4: -14.0147333, -10.0288086, -14.0153484, -10.0193348, -3.4153013, 3.4111362
5: -8.5682497, -5.0967379, -8.5588245, -5.0873432, -2.2796366, 2.2807908
6: -12.7698956, -8.5420132, -12.7833195, -8.5335722, -3.4231524, 3.4227719
7: -9.1891289, -5.7050209, -9.2204752, -5.6984711, -2.9482241, 2.9940395
8: 9.6494322, 12.5929327, 9.6281052, 12.5919476, -2.7591815, 2.7739410
9: -7.9703417, -3.7021971, -7.9758325, -3.6867185, -3.1397381, 3.1188068

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 4627

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6195301, upper bound: 1.6202181
time: 5.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6199269, upper bound: 1.6231038
time: 5.60 seconds

## BFS IS instance: IS_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -10.9750786, -6.9026737, -10.9661674, -6.9065037, -3.1373458, 3.1335258
1: -9.9472580, -6.8174038, -9.9400902, -6.8155432, -2.8211021, 2.8161979
2: -4.8396773, -1.7283592, -4.8190284, -1.7215557, -2.8694344, 2.8509278
3: -1.6622972, 1.6311388, -1.6598907, 1.6174273, -3.2797246, 3.2910295
4: -13.9685478, -10.0429478, -13.9504633, -10.0740414, -3.3001890, 3.3335824
5: -8.5294943, -5.1152277, -8.5236893, -5.1217980, -2.2256055, 2.2126272
6: -12.7450409, -8.5663595, -12.7421608, -8.5591249, -3.3645296, 3.3506966
7: -9.1637764, -5.7136955, -9.1647396, -5.7196193, -2.8773670, 2.9106288
8: 9.6739349, 12.5668564, 9.6715508, 12.5504770, -2.6878328, 2.7020264
9: -7.9344268, -3.7377436, -7.9387865, -3.7305217, -3.0259972, 3.0410781

Time for backsubstitution: 14.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 6253

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6193167, upper bound: 1.6122871
time: 5.61 seconds

## Relational analysis of IS_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6193167, upper bound: 1.6150036
time: 5.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 38.02 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6136534, upper bound: 1.6141868
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6136534, upper bound: 1.6174910
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6145810
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6178886
IS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6176576, upper bound: 1.6154697
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6176576, upper bound: 1.6187747
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6158606
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6191682
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6123744, upper bound: 1.6162186
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6127730, upper bound: 1.6191032
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6136586, upper bound: 1.6202186
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6140562, upper bound: 1.6231040
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6182459, upper bound: 1.6162182
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6186445, upper bound: 1.6191030
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6195301, upper bound: 1.6202181
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6199269, upper bound: 1.6231038
IS_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6193167, upper bound: 1.6122871
IS_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 38.02
Output dim: 8, lower bound: -1.6193167, upper bound: 1.6150036
IS_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 38.02
Output dim: 8, lower bound: -1.6197149, upper bound: 1.6178882
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 38.02
Output dim: 8, lower bound: -1.6168300, upper bound: 1.6187748
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 38.02
Output dim: 8, lower bound: -1.6237142, upper bound: 1.6191687
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 38.02
Output dim: 8, lower bound: -1.6159518, upper bound: 1.6191065
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 38.02
Output dim: 8, lower bound: -1.6172360, upper bound: 1.6231071
IS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 38.02
Output dim: 8, lower bound: -1.6191064, upper bound: 1.6218229
IS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 38.02
Output dim: 8, lower bound: -1.6231072, upper bound: 1.6231073
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7666475772857666
rel_dist={8: [-1.6237368538932522, 1.6237361751053658]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 6253
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952819, upper bound: 1.4930238
time: 12.27 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952819, upper bound: 1.4952810
time: 8.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 20.59 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 20.59
Output dim: 8, lower bound: -1.4952819, upper bound: 1.4930238
IS_B2, status: Status.UNKNOWN, split count: 1, time: 20.59
Output dim: 8, lower bound: -1.4952819, upper bound: 1.4952810

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -11.0299339, -6.8570595, -11.0289745, -6.8591261, -3.1093912, 3.1110601
1: -9.9668102, -6.7874885, -9.9610510, -6.7885365, -2.7918429, 2.7875094
2: -4.8408098, -1.6950119, -4.8398800, -1.7017744, -2.8322558, 2.8390422
3: -1.7237175, 1.6608000, -1.7212706, 1.6580515, -3.3817689, 3.3820705
4: -14.0008450, -10.0292339, -13.9953804, -10.0299664, -3.2829885, 3.2779093
5: -8.5538702, -5.0979548, -8.5511646, -5.0987401, -2.1619568, 2.1599288
6: -12.7711935, -8.5420647, -12.7698078, -8.5451317, -3.2889881, 3.2905378
7: -9.1901073, -5.7015324, -9.1839590, -5.7023129, -2.8390951, 2.8337569
8: 9.6456776, 12.5809994, 9.6516819, 12.5805035, -2.6733279, 2.6667399
9: -7.9725299, -3.7004180, -7.9719172, -3.7013087, -2.9957132, 2.9964943

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918482, upper bound: 1.4930215
time: 22.27 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4930208
time: 16.29 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -11.0312014, -6.8542767, -11.0377007, -6.8517303, -3.1244559, 3.1243005
1: -9.9745502, -6.7860970, -9.9825935, -6.7680578, -2.8194942, 2.8081870
2: -4.8420515, -1.6859413, -4.8699226, -1.6805701, -2.8484459, 2.8718593
3: -1.7269831, 1.6645001, -1.7423849, 1.6700652, -3.3970482, 3.4068851
4: -14.0081930, -10.0282612, -14.0116749, -10.0041485, -3.3167553, 3.2936316
5: -8.5575171, -5.0969095, -8.5616531, -5.0880113, -2.1761422, 2.1693203
6: -12.7730370, -8.5379372, -12.7881174, -8.5354910, -3.2967768, 3.3128376
7: -9.1983671, -5.7004967, -9.2025137, -5.6786847, -2.8710032, 2.8473601
8: 9.6376410, 12.5816555, 9.6287947, 12.5982733, -2.7011070, 2.6898413
9: -7.9733362, -3.6992116, -7.9776421, -3.6958699, -3.0030966, 3.0041785

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918482, upper bound: 1.4952794
time: 26.37 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4952784
time: 6.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 48.16 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 48.16
Output dim: 8, lower bound: -1.4918482, upper bound: 1.4930215
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 48.16
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4930208
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 48.16
Output dim: 8, lower bound: -1.4918482, upper bound: 1.4952794
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 48.16
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4952784

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -10.9930849, -6.8987532, -11.0238571, -6.8827548, -3.0520239, 3.0657387
1: -9.9423857, -6.7953253, -9.9572439, -6.7927899, -2.7607174, 2.7727742
2: -4.8268929, -1.7089313, -4.8341970, -1.7054816, -2.8131957, 2.8188915
3: -1.6850274, 1.6204882, -1.6998494, 1.6517212, -3.3367486, 3.3203375
4: -13.9621506, -10.0650978, -13.9887581, -10.0505257, -3.2270441, 3.2369552
5: -8.5343113, -5.1186700, -8.5486145, -5.1099234, -2.1304941, 2.1362739
6: -12.7603817, -8.5537138, -12.7667665, -8.5504627, -3.2686586, 3.2715688
7: -9.1709089, -5.7064896, -9.1751575, -5.7052865, -2.8157616, 2.8140073
8: 9.6634302, 12.5640783, 9.6598654, 12.5777187, -2.6507883, 2.6402063
9: -7.9499078, -3.7145298, -7.9650517, -3.7081127, -2.9651318, 2.9749041

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_B1_A1_B1

### Relational analysis result of IS_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906118, upper bound: 1.4897179
time: 9.47 seconds

## Relational analysis of IS_B1_A1_B2

### Relational analysis result of IS_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918422, upper bound: 1.4930144
time: 6.46 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -11.0299263, -6.8570890, -11.0289717, -6.8591423, -3.1093645, 3.0749030
1: -9.9668045, -6.7874947, -9.9610481, -6.7885408, -2.7918291, 2.7851262
2: -4.8408012, -1.6950197, -4.8398738, -1.7017782, -2.8277097, 2.8390288
3: -1.7236814, 1.6607888, -1.7212491, 1.6580458, -3.3817272, 3.3820379
4: -14.0008316, -10.0292549, -13.9953718, -10.0299778, -3.2829618, 3.2684131
5: -8.5538645, -5.0979757, -8.5511627, -5.0987520, -2.1619420, 2.1399317
6: -12.7711906, -8.5420723, -12.7698030, -8.5451365, -3.2778230, 3.2965393
7: -9.1900921, -5.7015362, -9.1839495, -5.7023163, -2.8486061, 2.8215790
8: 9.6456871, 12.5809956, 9.6516857, 12.5805016, -2.6648302, 2.6667275
9: -7.9725204, -3.7004280, -7.9719100, -3.7013152, -2.9988694, 2.9952979

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946954, upper bound: 1.4880253
time: 8.30 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946953, upper bound: 1.4924374
time: 6.09 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -10.9943485, -6.8959670, -11.0325794, -6.8753543, -3.0670853, 3.0789752
1: -9.9501400, -6.7939301, -9.9787693, -6.7723098, -2.7871857, 2.7934294
2: -4.8281298, -1.6998584, -4.8642459, -1.6842812, -2.8293829, 2.8510127
3: -1.6883104, 1.6242009, -1.7209897, 1.6637623, -3.3520727, 3.3451905
4: -13.9695110, -10.0641222, -14.0050774, -10.0247059, -3.2607956, 3.2526417
5: -8.5379543, -5.1176276, -8.5591030, -5.0991921, -2.1446905, 2.1456511
6: -12.7622185, -8.5495815, -12.7850676, -8.5408173, -3.2764406, 3.2938442
7: -9.1791821, -5.7054539, -9.1937227, -5.6816602, -2.8476410, 2.8275723
8: 9.6553841, 12.5647354, 9.6369715, 12.5954819, -2.6785669, 2.6633091
9: -7.9507089, -3.7133224, -7.9707479, -3.7026713, -2.9725142, 2.9825587

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906118, upper bound: 1.4919754
time: 10.68 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918422, upper bound: 1.4952735
time: 9.31 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -11.0311956, -6.8543077, -11.0376987, -6.8517475, -3.1244278, 3.0881438
1: -9.9745455, -6.7861028, -9.9825897, -6.7680626, -2.8131418, 2.8058019
2: -4.8420429, -1.6859481, -4.8699174, -1.6805743, -2.8439021, 2.8679566
3: -1.7269540, 1.6644895, -1.7423668, 1.6700581, -3.3970122, 3.4068563
4: -14.0081835, -10.0282822, -14.0116692, -10.0041599, -3.3026552, 3.2841334
5: -8.5575123, -5.0969291, -8.5616484, -5.0880227, -2.1730080, 2.1493230
6: -12.7730360, -8.5379467, -12.7881193, -8.5354958, -3.2856317, 3.3188224
7: -9.1983538, -5.7005000, -9.2025051, -5.6786857, -2.8805113, 2.8352127
8: 9.6376524, 12.5816507, 9.6288004, 12.5982695, -2.6926103, 2.6898270
9: -7.9733248, -3.6992211, -7.9776363, -3.6958761, -3.0062523, 3.0029831

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946954, upper bound: 1.4902822
time: 8.61 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946954, upper bound: 1.4946952
time: 6.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 29.95 seconds
IS_B1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4906118, upper bound: 1.4897179
IS_B1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4918422, upper bound: 1.4930144
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4946954, upper bound: 1.4880253
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4946953, upper bound: 1.4924374
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4906118, upper bound: 1.4919754
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4918422, upper bound: 1.4952735
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4946954, upper bound: 1.4902822
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.95
Output dim: 8, lower bound: -1.4946954, upper bound: 1.4946952

## BFS IS instance: IS_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.9727230, -6.9076414, -10.9895029, -6.8982143, -3.0063324, 3.0109091
1: -9.9373140, -6.8068409, -9.9485550, -6.8122468, -2.7345314, 2.7508216
2: -4.8240614, -1.7182562, -4.8293648, -1.7211999, -2.7876043, 2.7961040
3: -1.6764765, 1.6171958, -1.6854138, 1.6462166, -3.3226931, 3.3026097
4: -13.9576855, -10.0741997, -13.9811306, -10.0658398, -3.2019625, 3.2143250
5: -8.5279922, -5.1205549, -8.5379391, -5.1131368, -2.1195865, 2.1220164
6: -12.7519388, -8.5626869, -12.7529879, -8.5655193, -3.2441311, 3.2472882
7: -9.1662178, -5.7208118, -9.1670380, -5.7295017, -2.7832270, 2.7879305
8: 9.6751146, 12.5622234, 9.6796188, 12.5745602, -2.6324039, 2.6160817
9: -7.9423208, -3.7179391, -7.9522772, -3.7139189, -2.9499550, 2.9570007

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_B1_A1_B1_B1

### Relational analysis result of IS_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881878, upper bound: 1.4891461
time: 9.18 seconds

## Relational analysis of IS_B1_A1_B1_B2

### Relational analysis result of IS_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906093, upper bound: 1.4897159
time: 12.55 seconds

## BFS IS instance: IS_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.9930391, -6.8987646, -11.0282440, -6.8498998, -3.0895076, 3.0615401
1: -9.9423771, -6.7953525, -9.9863911, -6.7899356, -2.7600074, 2.7989492
2: -4.8268886, -1.7089542, -4.8499508, -1.7024696, -2.8140121, 2.8290606
3: -1.6850064, 1.6204828, -1.7169554, 1.6617647, -3.3467712, 3.3374381
4: -13.9621420, -10.0651169, -14.0018616, -10.0408497, -3.2323847, 3.2455864
5: -8.5342960, -5.1186738, -8.5529165, -5.0994802, -2.1402869, 2.1397617
6: -12.7603683, -8.5537357, -12.7795258, -8.5427513, -3.2750921, 3.2842660
7: -9.1709042, -5.7065191, -9.2038507, -5.6990781, -2.8170280, 2.8401380
8: 9.6634541, 12.5640764, 9.6437197, 12.5885458, -2.6603293, 2.6513591
9: -7.9498916, -3.7145352, -7.9703860, -3.6946390, -2.9797230, 2.9808693

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_B1_A1_B2_A1

### Relational analysis result of IS_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4912669, upper bound: 1.4905898
time: 9.82 seconds

## Relational analysis of IS_B1_A1_B2_A2

### Relational analysis result of IS_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918399, upper bound: 1.4930124
time: 14.25 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0293694, -6.8594460, -11.0279541, -6.8634396, -3.1020565, 3.0700154
1: -9.9659634, -6.7911062, -9.9595127, -6.7951241, -2.7832856, 2.7791901
2: -4.8376036, -1.6956447, -4.8340502, -1.7029225, -2.8233366, 2.8324347
3: -1.7184877, 1.6584692, -1.7117724, 1.6538050, -3.3722928, 3.3702416
4: -13.9983244, -10.0363989, -13.9907856, -10.0429974, -3.2678967, 3.2581038
5: -8.5531883, -5.1035872, -8.5499287, -5.1089816, -2.1511521, 2.1332879
6: -12.7705975, -8.5439434, -12.7687225, -8.5485544, -3.2740898, 3.2902470
7: -9.1882067, -5.7022767, -9.1805058, -5.7036672, -2.8392115, 2.8177061
8: 9.6497555, 12.5802240, 9.6590996, 12.5790930, -2.6594949, 2.6585972
9: -7.9712172, -3.7008851, -7.9695315, -3.7021511, -2.9943428, 2.9912491

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913910, upper bound: 1.4867876
time: 9.22 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946890, upper bound: 1.4880191
time: 6.79 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.0299129, -6.8570957, -11.0363235, -6.8579192, -3.1041150, 3.0819993
1: -9.9668016, -6.7875490, -9.9828882, -6.7877159, -2.7882080, 2.8024197
2: -4.8407912, -1.6950765, -4.8436966, -1.6918392, -2.8378477, 2.8400779
3: -1.7236004, 1.6607778, -1.7239718, 1.6753888, -3.3989892, 3.3847497
4: -14.0008259, -10.0292921, -14.0147409, -10.0287981, -3.2810917, 3.2842951
5: -8.5538197, -5.0979948, -8.5682592, -5.0967388, -2.1587658, 2.1551392
6: -12.7700043, -8.5420818, -12.7699070, -8.5420494, -3.2863674, 3.2925382
7: -9.1900854, -5.7048960, -9.1886139, -5.7049923, -2.8481140, 2.8275485
8: 9.6458206, 12.5809908, 9.6494141, 12.5929346, -2.6775370, 2.6668947
9: -7.9699793, -3.7004492, -7.9703526, -3.7021918, -2.9912372, 3.0034642

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913909, upper bound: 1.4912004
time: 9.21 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946890, upper bound: 1.4924313
time: 7.48 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.9739895, -6.9048624, -10.9982204, -6.8908157, -3.0212164, 3.0241899
1: -9.9450684, -6.8054485, -9.9700851, -6.7917767, -2.7610512, 2.7714500
2: -4.8252969, -1.7091743, -4.8594079, -1.6999687, -2.8037834, 2.8265898
3: -1.6797450, 1.6209102, -1.7064650, 1.6582699, -3.3380148, 3.3273752
4: -13.9650517, -10.0732241, -13.9974775, -10.0400219, -3.2357044, 3.2299824
5: -8.5316353, -5.1195145, -8.5484495, -5.1023993, -2.1337876, 2.1313620
6: -12.7537842, -8.5585527, -12.7711678, -8.5558596, -3.2519236, 3.2695212
7: -9.1744900, -5.7197790, -9.1855764, -5.7058883, -2.8150883, 2.8014698
8: 9.6670618, 12.5628805, 9.6566696, 12.5923233, -2.6601872, 2.6391349
9: -7.9431238, -3.7167244, -7.9579706, -3.7084587, -2.9573312, 2.9646516

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_B2_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881878, upper bound: 1.4914055
time: 7.13 seconds

## Relational analysis of IS_B2_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4906093, upper bound: 1.4919733
time: 7.54 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.9943066, -6.8959799, -11.0369682, -6.8424811, -3.0995736, 3.0747824
1: -9.9501324, -6.7939591, -10.0079126, -6.7694421, -2.7865601, 2.8198216
2: -4.8281255, -1.6998799, -4.8800125, -1.6812503, -2.8301644, 2.8536668
3: -1.6882904, 1.6241956, -1.7380745, 1.6737665, -3.3620567, 3.3622701
4: -13.9695053, -10.0641403, -14.0181665, -10.0150280, -3.2661219, 3.2613072
5: -8.5379372, -5.1176319, -8.5634232, -5.0887518, -2.1544814, 2.1491356
6: -12.7622061, -8.5496044, -12.7977962, -8.5331240, -3.2828684, 3.3065171
7: -9.1791744, -5.7054868, -9.2223959, -5.6754599, -2.8488979, 2.8537550
8: 9.6554079, 12.5647287, 9.6207848, 12.6063213, -2.6872931, 2.6744289
9: -7.9506950, -3.7133317, -7.9760885, -3.6892056, -2.9870954, 2.9885325

Time for backsubstitution: 14.25 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4894154, upper bound: 1.4946982
time: 6.87 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918397, upper bound: 1.4952710
time: 15.05 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.0306358, -6.8566651, -11.0366726, -6.8560467, -3.1171188, 3.0832534
1: -9.9737024, -6.7897139, -9.9810390, -6.7746453, -2.8045740, 2.7998743
2: -4.8388462, -1.6865730, -4.8640971, -1.6817203, -2.8395276, 2.8613782
3: -1.7217660, 1.6621716, -1.7329123, 1.6658375, -3.3876035, 3.3950839
4: -14.0056744, -10.0354252, -14.0071030, -10.0171785, -3.2875166, 3.2737999
5: -8.5568361, -5.1025429, -8.5604162, -5.0982475, -2.1621685, 2.1426783
6: -12.7724419, -8.5398178, -12.7870369, -8.5389099, -3.2818832, 3.3125253
7: -9.1964703, -5.7012405, -9.1990700, -5.6800365, -2.8711109, 2.8313079
8: 9.6417189, 12.5808811, 9.6362152, 12.5968561, -2.6872678, 2.6816988
9: -7.9720221, -3.6996810, -7.9752421, -3.6967177, -3.0017271, 2.9989257

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
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

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913910, upper bound: 1.4890444
time: 6.14 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4902763
time: 23.54 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.0311823, -6.8543143, -11.0450439, -6.8505239, -3.1191797, 3.0952368
1: -9.9745417, -6.7861567, -10.0044003, -6.7672472, -2.8100195, 2.8232293
2: -4.8420324, -1.6860037, -4.8737416, -1.6706374, -2.8540392, 2.8690526
3: -1.7268739, 1.6644771, -1.7450838, 1.6873758, -3.4142497, 3.4095609
4: -14.0081720, -10.0283203, -14.0310259, -10.0029812, -3.3007917, 3.2961936
5: -8.5574646, -5.0969505, -8.5787487, -5.0860200, -2.1697977, 2.1640427
6: -12.7718487, -8.5379534, -12.7882423, -8.5324345, -3.2941513, 3.3148336
7: -9.1983471, -5.7038636, -9.2071323, -5.6813650, -2.8800211, 2.8411732
8: 9.6377831, 12.5816498, 9.6265430, 12.6106968, -2.6974776, 2.6899993
9: -7.9707856, -3.6992466, -7.9760675, -3.6967692, -2.9986181, 3.0111432

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
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

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4627

## Relational analysis of IS_B2_A2_B2_B1

### Relational analysis result of IS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4934586, upper bound: 1.4913905
time: 9.89 seconds

## Relational analysis of IS_B2_A2_B2_B2

### Relational analysis result of IS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946893, upper bound: 1.4946885
time: 6.55 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.07 seconds
IS_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4881878, upper bound: 1.4891461
IS_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4906093, upper bound: 1.4897159
IS_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4912669, upper bound: 1.4905898
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4918399, upper bound: 1.4930124
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4913910, upper bound: 1.4867876
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4946890, upper bound: 1.4880191
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4913909, upper bound: 1.4912004
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4946890, upper bound: 1.4924313
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4881878, upper bound: 1.4914055
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4906093, upper bound: 1.4919733
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4894154, upper bound: 1.4946982
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4918397, upper bound: 1.4952710
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4913910, upper bound: 1.4890444
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4902763
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4934586, upper bound: 1.4913905
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 31.07
Output dim: 8, lower bound: -1.4946893, upper bound: 1.4946885

## BFS IS instance: IS_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -10.9592628, -6.9115105, -10.9655285, -6.9138007, -2.9749098, 2.9879899
1: -9.9302969, -6.8211608, -9.9250231, -6.8385673, -2.7004805, 2.7093744
2: -4.8161592, -1.7347449, -4.8087721, -1.7501948, -2.7526498, 2.7583284
3: -1.6512282, 1.6123853, -1.6378253, 1.6180186, -3.2563715, 3.2502105
4: -13.9396305, -10.0770464, -13.9511786, -10.0720243, -3.1736612, 3.1592979
5: -8.5172873, -5.1236587, -8.5185566, -5.1277480, -2.0762153, 2.0977576
6: -12.7365627, -8.5652180, -12.7263985, -8.5768776, -3.2158971, 3.2179141
7: -9.1537895, -5.7236018, -9.1438580, -5.7377501, -2.7596369, 2.7422252
8: 9.6828756, 12.5472050, 9.6981773, 12.5486345, -2.5881534, 2.5771646
9: -7.9355412, -3.7349575, -7.9276233, -3.7442935, -2.9092121, 2.8924046

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4630
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 6253

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_B1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4861232, upper bound: 1.4891452
time: 5.33 seconds

## Relational analysis of IS_B1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4861232, upper bound: 1.4891462
time: 8.69 seconds

## BFS IS instance: IS_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -10.9727173, -6.9076443, -10.9894934, -6.8982182, -3.0081134, 3.0104775
1: -9.9373112, -6.8068509, -9.9485512, -6.8122625, -2.7339025, 2.7534304
2: -4.8240590, -1.7182667, -4.8293614, -1.7212193, -2.7875853, 2.7962990
3: -1.6764684, 1.6171949, -1.6853976, 1.6462139, -3.3226824, 3.3025925
4: -13.9576807, -10.0741997, -13.9811230, -10.0658426, -3.1853724, 3.2379942
5: -8.5279884, -5.1205573, -8.5379362, -5.1131368, -2.1195803, 2.1191373
6: -12.7519360, -8.5626888, -12.7529793, -8.5655193, -3.2441235, 3.2386465
7: -9.1662140, -5.7208109, -9.1670332, -5.7295032, -2.7759266, 2.7944055
8: 9.6751175, 12.5622196, 9.6796227, 12.5745516, -2.6260457, 2.6097150
9: -7.9423199, -3.7179432, -7.9522748, -3.7139311, -2.9478378, 2.9569883

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 5762
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832
type: B, layer: 1, pos: 832
type: A, layer: 1, pos: 6253

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of IS_B1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881135, upper bound: 1.4897158
time: 26.17 seconds

## Relational analysis of IS_B1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881135, upper bound: 1.4897160
time: 9.56 seconds

## BFS IS instance: IS_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.9690599, -6.9143767, -11.0147877, -6.8537722, -3.0667825, 3.0301380
1: -9.9188337, -6.8216696, -9.9793758, -6.8042536, -2.7186260, 2.7648456
2: -4.8062973, -1.7379506, -4.8420439, -1.7189543, -2.7762594, 2.7940459
3: -1.6374593, 1.5922415, -1.6916697, 1.6570079, -3.2944672, 3.2839112
4: -13.9321547, -10.0712910, -13.9838152, -10.0437021, -3.1773729, 3.2172680
5: -8.5149393, -5.1333065, -8.5422115, -5.1025639, -2.1160498, 2.0962994
6: -12.7338009, -8.5650768, -12.7641363, -8.5452757, -3.2456965, 3.2560616
7: -9.1477423, -5.7147722, -9.1913891, -5.7018390, -2.7713928, 2.8165016
8: 9.6820621, 12.5381413, 9.6514072, 12.5735321, -2.6214318, 2.6071987
9: -7.9252453, -3.7448978, -7.9636278, -3.7116570, -2.9151182, 2.9401455

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
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
type: B, layer: 1, pos: 6253
type: A, layer: 1, pos: 4627

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of IS_B1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4887715, upper bound: 1.4905899
time: 7.88 seconds

## Relational analysis of IS_B1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4887715, upper bound: 1.4905902
time: 7.43 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 42.61 seconds
IS_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 8, lower bound: -1.4861232, upper bound: 1.4891452
IS_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 8, lower bound: -1.4861232, upper bound: 1.4891462
IS_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 8, lower bound: -1.4881135, upper bound: 1.4897158
IS_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 8, lower bound: -1.4881135, upper bound: 1.4897160
IS_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 8, lower bound: -1.4887715, upper bound: 1.4905899
IS_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 42.61
Output dim: 8, lower bound: -1.4887715, upper bound: 1.4905902
IS_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4918399, upper bound: 1.4930124
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4913910, upper bound: 1.4867876
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4946890, upper bound: 1.4880191
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4913909, upper bound: 1.4912004
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4946890, upper bound: 1.4924313
IS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4881878, upper bound: 1.4914055
IS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4906093, upper bound: 1.4919733
IS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4894154, upper bound: 1.4946982
IS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4918397, upper bound: 1.4952710
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4913910, upper bound: 1.4890444
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4902763
IS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4934586, upper bound: 1.4913905
IS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 42.61
Output dim: 8, lower bound: -1.4946893, upper bound: 1.4946885
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.6844100952148438
rel_dist={8: [-1.4952889722192388, 1.4952881994988214]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2418.73 seconds
