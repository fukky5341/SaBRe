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
execution time: IAR + LP analysis = 15.20 + 35.00 = 50.19 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -2.4068807, upper bound: 2.4068790


# Binary Search by BASE starts (time budget: 3549.81 seconds, max iter: 100)

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
Binary search time: 217.70 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3332.11 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9688919
time: 4.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688918, upper bound: 1.9688919
time: 4.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.75
Output dim: 8, lower bound: -1.9632782, upper bound: 1.9688919
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.75
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

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9631922, upper bound: 1.9621188
time: 4.94 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632739, upper bound: 1.9688876
time: 6.30 seconds

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

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688058, upper bound: 1.9621186
time: 5.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688875, upper bound: 1.9688875
time: 5.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.55 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 8, lower bound: -1.9631922, upper bound: 1.9621188
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 8, lower bound: -1.9632739, upper bound: 1.9688876
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 8, lower bound: -1.9688058, upper bound: 1.9621186
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.55
Output dim: 8, lower bound: -1.9688875, upper bound: 1.9688875

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -11.0265465, -6.8704195, -10.9939280, -6.8969040, -3.6015763, 3.5965576
1: -9.9592409, -6.7905712, -9.9475422, -6.7943964, -3.1168599, 3.1093340
2: -4.8371596, -1.7035393, -4.8277164, -1.7028959, -3.1342638, 3.1241770
3: -1.7110248, 1.6550372, -1.6872125, 1.6229558, -3.3339806, 3.3422496
4: -13.9922333, -10.0397949, -13.9670439, -10.0644474, -3.7238293, 3.7274437
5: -8.5499535, -5.1040878, -8.5367346, -5.1179743, -2.6051941, 2.6059465
6: -12.7683582, -8.5476904, -12.7616024, -8.5509663, -3.7640386, 3.7597947
7: -9.1797352, -5.7037401, -9.1764107, -5.7057981, -3.2832575, 3.2902956
8: 9.6555929, 12.5791807, 9.6580725, 12.5645161, -2.9089231, 2.9211082
9: -7.9686475, -3.7045641, -7.9504418, -3.7137270, -3.4617009, 3.4523296

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9571835, upper bound: 1.9612492
time: 5.48 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9631808, upper bound: 1.9621077
time: 5.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -11.0289745, -6.8591318, -11.0307722, -6.8552384, -3.6156549, 3.6417561
1: -9.9610491, -6.7885385, -9.9719534, -6.7865667, -3.1274118, 3.1375523
2: -4.8398762, -1.7017765, -4.8416281, -1.6889833, -3.1508927, 3.1398516
3: -1.7212610, 1.6580493, -1.7258577, 1.6632488, -3.3845098, 3.3839071
4: -13.9953756, -10.0299683, -14.0057182, -10.0286074, -3.7549715, 3.7722330
5: -8.5511637, -5.0987434, -8.5562897, -5.0972805, -2.6114278, 2.6314566
6: -12.7698021, -8.5451317, -12.7724209, -8.5393286, -3.7906213, 3.7695746
7: -9.1839533, -5.7023168, -9.1955862, -5.7008462, -3.2891417, 3.3262839
8: 9.6516829, 12.5805025, 9.6403351, 12.5814333, -2.9297504, 2.9401674
9: -7.9719143, -3.7013113, -7.9730597, -3.6996250, -3.4793844, 3.4822865

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9572652, upper bound: 1.9680167
time: 8.67 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632626, upper bound: 1.9688764
time: 5.50 seconds

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

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627972, upper bound: 1.9612496
time: 5.91 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687943, upper bound: 1.9621076
time: 5.77 seconds

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9628789, upper bound: 1.9680170
time: 6.40 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688761, upper bound: 1.9688764
time: 5.45 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.55 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9571835, upper bound: 1.9612492
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9631808, upper bound: 1.9621077
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9572652, upper bound: 1.9680167
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9632626, upper bound: 1.9688764
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9627972, upper bound: 1.9612496
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9687943, upper bound: 1.9621076
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9628789, upper bound: 1.9680170
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.55
Output dim: 8, lower bound: -1.9688761, upper bound: 1.9688764

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.9921980, -6.8858862, -10.9857082, -6.9004025, -3.5538702, 3.5660152
1: -9.9505596, -6.8100257, -9.9455233, -6.7990437, -3.1021171, 3.0866160
2: -4.8323212, -1.7192637, -4.8265834, -1.7066654, -3.1246772, 3.1065788
3: -1.6966324, 1.6495383, -1.6837482, 1.6216247, -3.3182571, 3.3332865
4: -13.9845867, -10.0550690, -13.9652615, -10.0681305, -3.7075510, 3.7065783
5: -8.5392780, -5.1072984, -8.5341797, -5.1187353, -2.5922508, 2.5992961
6: -12.7545586, -8.5627327, -12.7580547, -8.5546064, -3.7452669, 3.7406445
7: -9.1715860, -5.7279592, -9.1745501, -5.7115774, -3.2665339, 3.2613611
8: 9.6753445, 12.5760145, 9.6627846, 12.5637741, -2.8884296, 2.9132299
9: -7.9558716, -3.7103703, -7.9473758, -3.7150853, -3.4462671, 3.4420166

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9571836, upper bound: 1.9556355
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9571836, upper bound: 1.9612492
time: 4.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.0309362, -6.8375587, -10.9939089, -6.8969083, -3.5974045, 3.6421494
1: -9.9883823, -6.7877183, -9.9475384, -6.7944093, -3.1458344, 3.1107850
2: -4.8529263, -1.7005261, -4.8277159, -1.7029071, -3.1500192, 3.1271896
3: -1.7281432, 1.6650636, -1.6872032, 1.6229546, -3.3510978, 3.3522668
4: -14.0053358, -10.0300703, -13.9670410, -10.0644550, -3.7325058, 3.7355695
5: -8.5542507, -5.0936413, -8.5367250, -5.1179786, -2.6093178, 2.6157458
6: -12.7811165, -8.5399780, -12.7615986, -8.5509777, -3.7767534, 3.7666283
7: -9.2084818, -5.6975365, -9.1764088, -5.7058144, -3.3094330, 3.2951097
8: 9.6394548, 12.5900049, 9.6580811, 12.5645161, -2.9250612, 2.9319239
9: -7.9739766, -3.6910784, -7.9504356, -3.7137308, -3.4676647, 3.4669752

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9631808, upper bound: 1.9564940
time: 5.24 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9631808, upper bound: 1.9621077
time: 5.00 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.9946327, -6.8746014, -11.0225658, -6.8587384, -3.5679493, 3.6112041
1: -9.9523754, -6.8079915, -9.9699402, -6.7912145, -3.1126690, 3.1148367
2: -4.8350334, -1.7175038, -4.8404913, -1.6927599, -3.1415997, 3.1198959
3: -1.7068903, 1.6525555, -1.7224219, 1.6619256, -3.3688159, 3.3749774
4: -13.9877100, -10.0451994, -14.0039291, -10.0322819, -3.7386923, 3.7513580
5: -8.5404892, -5.1019621, -8.5537357, -5.0980396, -2.5984831, 2.6247997
6: -12.7560034, -8.5601654, -12.7688503, -8.5429592, -3.7718844, 3.7504315
7: -9.1757832, -5.7265358, -9.1937027, -5.7066264, -3.2723684, 3.2973108
8: 9.6714334, 12.5773373, 9.6450462, 12.5806904, -2.9092569, 2.9322910
9: -7.9591331, -3.7071195, -7.9699931, -3.7009826, -3.4639244, 3.4719534

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9572652, upper bound: 1.9624029
time: 27.96 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9572652, upper bound: 1.9680165
time: 10.70 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.0333700, -6.8262625, -11.0307493, -6.8552446, -3.6114969, 3.6722052
1: -9.9901867, -6.7856865, -9.9719486, -6.7865806, -3.1563869, 3.1390014
2: -4.8556595, -1.6987630, -4.8416262, -1.6889943, -3.1666651, 3.1428633
3: -1.7383738, 1.6680586, -1.7258503, 1.6632460, -3.4016199, 3.3939090
4: -14.0084820, -10.0202084, -14.0057135, -10.0286140, -3.7636766, 3.7803588
5: -8.5554590, -5.0883007, -8.5562811, -5.0972791, -2.6155467, 2.6412528
6: -12.7825661, -8.5374203, -12.7724113, -8.5393391, -3.8033543, 3.7763991
7: -9.2127581, -5.6961088, -9.1955833, -5.7008605, -3.3155670, 3.3311152
8: 9.6355495, 12.5913286, 9.6403465, 12.5814323, -2.9458828, 2.9509821
9: -7.9772363, -3.6878130, -7.9730511, -3.6996312, -3.4853153, 3.4969459

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632626, upper bound: 1.9632625
time: 5.60 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9632626, upper bound: 1.9688764
time: 5.95 seconds

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

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627970, upper bound: 1.9556359
time: 5.52 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627974, upper bound: 1.9556359
time: 4.94 seconds

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

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687942, upper bound: 1.9564939
time: 5.35 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687945, upper bound: 1.9564940
time: 5.13 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0033550, -6.8672056, -11.0229893, -6.8578053, -3.5780168, 3.6268692
1: -9.9739227, -6.7875237, -9.9725370, -6.7907491, -3.1340685, 3.1357644
2: -4.8650703, -1.6962693, -4.8409066, -1.6897156, -3.1654558, 3.1415291
3: -1.7279391, 1.6645803, -1.7235131, 1.6631693, -3.3911085, 3.3880935
4: -14.0040369, -10.0193806, -14.0063925, -10.0319576, -3.7587643, 3.7715721
5: -8.5510006, -5.0912318, -8.5549622, -5.0976920, -2.6088486, 2.6365848
6: -12.7742033, -8.5505085, -12.7694664, -8.5415754, -3.7913971, 3.7590094
7: -9.1943159, -5.7029181, -9.1964731, -5.7062788, -3.2898293, 3.3233562
8: 9.6484890, 12.5951071, 9.6423521, 12.5809116, -2.9324226, 2.9527550
9: -7.9648380, -3.7016644, -7.9702611, -3.7005785, -3.4703484, 3.4793992

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9628788, upper bound: 1.9624028
time: 4.68 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9628791, upper bound: 1.9624030
time: 4.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.0420990, -6.8188472, -11.0311737, -6.8543072, -3.6215606, 3.6829906
1: -10.0117283, -6.7651930, -9.9745464, -6.7861147, -3.1779451, 3.1599824
2: -4.8857164, -1.6775417, -4.8420415, -1.6859521, -3.1858797, 3.1644998
3: -1.7594950, 1.6800327, -1.7269468, 1.6644888, -3.4239838, 3.4069796
4: -14.0247650, -9.9943886, -14.0081825, -10.0282898, -3.7838192, 3.8009043
5: -8.5659599, -5.0775776, -8.5575066, -5.0969296, -2.6259398, 2.6495011
6: -12.8008480, -8.5277967, -12.7730274, -8.5379534, -3.8228869, 3.7849565
7: -9.2313080, -5.6724854, -9.1983566, -5.7005138, -3.3331060, 3.3571706
8: 9.6126194, 12.6091061, 9.6376534, 12.5816507, -2.9690313, 2.9714527
9: -7.9829512, -3.6823847, -7.9733162, -3.6992266, -3.4917593, 3.5043883

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5845

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688759, upper bound: 1.9632625
time: 4.85 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9688763, upper bound: 1.9632625
time: 5.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.58 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9571836, upper bound: 1.9556355
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9571836, upper bound: 1.9612492
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9631808, upper bound: 1.9564940
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9631808, upper bound: 1.9621077
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9572652, upper bound: 1.9624029
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9572652, upper bound: 1.9680165
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9632626, upper bound: 1.9632625
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9632626, upper bound: 1.9688764
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9627970, upper bound: 1.9556359
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9627974, upper bound: 1.9556359
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9687942, upper bound: 1.9564939
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9687945, upper bound: 1.9564940
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9628788, upper bound: 1.9624028
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9628791, upper bound: 1.9624030
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9688759, upper bound: 1.9632625
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.58
Output dim: 8, lower bound: -1.9688763, upper bound: 1.9632625

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.9921980, -6.8858862, -10.9839087, -6.9043241, -3.5470333, 3.5624065
1: -9.9505596, -6.8100257, -9.9345970, -6.8010211, -3.0997014, 3.0760517
2: -4.8323212, -1.7192637, -4.8248320, -1.7194712, -3.1099644, 3.1046939
3: -1.6966324, 1.6495383, -1.6791186, 1.6163979, -3.3130302, 3.3286569
4: -13.9845867, -10.0550690, -13.9548883, -10.0695124, -3.7040882, 3.6935554
5: -8.5392780, -5.1072984, -8.5290575, -5.1202106, -2.5909672, 2.5941706
6: -12.7545586, -8.5627327, -12.7554522, -8.5604267, -3.7394948, 3.7378254
7: -9.1715860, -5.7279592, -9.1628895, -5.7130504, -3.2641888, 3.2489762
8: 9.6753445, 12.5760145, 9.6741562, 12.5628433, -2.8874989, 2.9018583
9: -7.9558716, -3.7103703, -7.9462318, -3.7167845, -3.4436321, 3.4408751

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9562620, upper bound: 1.9488607
time: 5.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9571791, upper bound: 1.9556310
time: 4.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.9921980, -6.8858862, -10.9926262, -6.8969135, -3.5548840, 3.5708513
1: -9.9505596, -6.8100257, -9.9561443, -6.7805409, -3.1220675, 3.0980239
2: -4.8323212, -1.7192637, -4.8548784, -1.6982555, -3.1301012, 3.1296377
3: -1.6966324, 1.6495383, -1.7002244, 1.6284927, -3.3251252, 3.3497627
4: -13.9845867, -10.0550690, -13.9712477, -10.0436945, -3.7286215, 3.7093782
5: -8.5392780, -5.1072984, -8.5395498, -5.1094713, -2.6015453, 2.6045923
6: -12.7545586, -8.5627327, -12.7736778, -8.5507727, -3.7488565, 3.7559690
7: -9.1715860, -5.7279592, -9.1814651, -5.6894279, -3.2873039, 3.2669387
8: 9.6753445, 12.5760145, 9.6512241, 12.5806084, -2.9052639, 2.9247904
9: -7.9558716, -3.7103703, -7.9519272, -3.7113175, -3.4487696, 3.4466648

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9562620, upper bound: 1.9544911
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9571791, upper bound: 1.9612447
time: 5.10 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0309362, -6.8375587, -10.9921064, -6.9008269, -3.5906429, 3.6385407
1: -9.9883823, -6.7877183, -9.9366131, -6.7963881, -3.1434197, 3.1002264
2: -4.8529263, -1.7005261, -4.8259621, -1.7157066, -3.1364031, 3.1254358
3: -1.7281432, 1.6650636, -1.6825564, 1.6177263, -3.3458695, 3.3476200
4: -14.0053358, -10.0300703, -13.9566708, -10.0658407, -3.7290430, 3.7225456
5: -8.5542507, -5.0936413, -8.5316038, -5.1194544, -2.6080346, 2.6106200
6: -12.7811165, -8.5399780, -12.7589903, -8.5567942, -3.7709832, 3.7638044
7: -9.2084818, -5.6975365, -9.1647472, -5.7072840, -3.3070908, 3.2827258
8: 9.6394548, 12.5900049, 9.6694489, 12.5635834, -2.9241285, 2.9205561
9: -7.9739766, -3.6910784, -7.9492912, -3.7154269, -3.4650292, 3.4658332

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9622674, upper bound: 1.9497377
time: 6.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9631764, upper bound: 1.9564894
time: 5.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.0309362, -6.8375587, -11.0008259, -6.8934197, -3.5985813, 3.6443510
1: -9.9883823, -6.7877183, -9.9581585, -6.7759047, -3.1528821, 3.1221781
2: -4.8529263, -1.7005261, -4.8560114, -1.6945016, -3.1565213, 3.1525297
3: -1.7281432, 1.6650636, -1.7037055, 1.6298199, -3.3579631, 3.3687692
4: -14.0053358, -10.0300703, -13.9730225, -10.0400181, -3.7535791, 3.7383647
5: -8.5542507, -5.0936413, -8.5420904, -5.1087127, -2.6186113, 2.6210375
6: -12.7811165, -8.5399780, -12.7772369, -8.5471449, -3.7803407, 3.7819643
7: -9.2084818, -5.6975365, -9.1833286, -5.6836605, -3.3302088, 3.3006945
8: 9.6394548, 12.5900049, 9.6465311, 12.5813503, -2.9418955, 2.9434738
9: -7.9739766, -3.6910784, -7.9549861, -3.7099614, -3.4701610, 3.4716234

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9622675, upper bound: 1.9553683
time: 5.47 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9631764, upper bound: 1.9621030
time: 4.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.9946327, -6.8746014, -11.0207624, -6.8626547, -3.5611124, 3.6075916
1: -9.9523754, -6.8079915, -9.9590321, -6.7931895, -3.1102533, 3.1042032
2: -4.8350334, -1.7175038, -4.8387327, -1.7055610, -3.1268849, 3.1180100
3: -1.7068903, 1.6525555, -1.7178016, 1.6567192, -3.3636096, 3.3703570
4: -13.9877100, -10.0451994, -13.9935722, -10.0336628, -3.7352333, 3.7383132
5: -8.5404892, -5.1019621, -8.5486069, -5.0995207, -2.5971928, 2.6196775
6: -12.7560034, -8.5601654, -12.7662354, -8.5487709, -3.7661004, 3.7476001
7: -9.1757832, -5.7265358, -9.1820602, -5.7080998, -3.2700224, 3.2848940
8: 9.6714334, 12.5773373, 9.6564083, 12.5797577, -2.9083242, 2.9209290
9: -7.9591331, -3.7071195, -7.9688416, -3.7026770, -3.4612870, 3.4708037

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9474155, upper bound: 1.9618146
time: 5.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9566756, upper bound: 1.9618149
time: 4.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.9946327, -6.8746014, -11.0294857, -6.8552589, -3.5689430, 3.6160426
1: -9.9523754, -6.8079915, -9.9805756, -6.7727141, -3.1312900, 3.1263828
2: -4.8350334, -1.7175038, -4.8687754, -1.6843481, -3.1470304, 3.1424005
3: -1.7068903, 1.6525555, -1.7388911, 1.6687340, -3.3756242, 3.3914466
4: -13.9877100, -10.0451994, -14.0098753, -10.0078430, -3.7597618, 3.7541142
5: -8.5404892, -5.1019621, -8.5590982, -5.0887928, -2.6077619, 2.6301038
6: -12.7560034, -8.5601654, -12.7845192, -8.5391293, -3.7754898, 3.7658072
7: -9.1757832, -5.7265358, -9.2006092, -5.6844721, -3.2931423, 3.3028946
8: 9.6714334, 12.5773373, 9.6335096, 12.5975256, -2.9260921, 2.9438276
9: -7.9591331, -3.7071195, -7.9745440, -3.6972332, -3.4663930, 3.4766216

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9474156, upper bound: 1.9674276
time: 5.25 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9566757, upper bound: 1.9674280
time: 4.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.0333700, -6.8262625, -11.0289469, -6.8591585, -3.6047368, 3.6688280
1: -9.9901867, -6.7856865, -9.9610424, -6.7885575, -3.1539707, 3.1283679
2: -4.8556595, -1.6987630, -4.8398676, -1.7017907, -3.1533527, 3.1411047
3: -1.7383738, 1.6680586, -1.7212248, 1.6580390, -3.3964128, 3.3892834
4: -14.0084820, -10.0202084, -13.9953642, -10.0299950, -3.7602167, 3.7673120
5: -8.5554590, -5.0883007, -8.5511532, -5.0987625, -2.6142554, 2.6361310
6: -12.7825661, -8.5374203, -12.7697964, -8.5451488, -3.7975721, 3.7735643
7: -9.2127581, -5.6961088, -9.1839380, -5.7023368, -3.3132229, 3.3186984
8: 9.6355495, 12.5913286, 9.6517038, 12.5804958, -2.9449463, 2.9396248
9: -7.9772363, -3.6878130, -7.9719000, -3.7013223, -3.4826794, 3.4957943

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9534202, upper bound: 1.9626768
time: 5.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626766, upper bound: 1.9626770
time: 4.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.0333700, -6.8262625, -11.0376730, -6.8517671, -3.6126614, 3.6727624
1: -9.9901867, -6.7856865, -9.9825830, -6.7680788, -3.1617174, 3.1505446
2: -4.8556595, -1.6987630, -4.8699121, -1.6805879, -3.1734829, 3.1652732
3: -1.7383738, 1.6680586, -1.7423458, 1.6700506, -3.4084244, 3.4104044
4: -14.0084820, -10.0202084, -14.0116615, -10.0041790, -3.7847471, 3.7831073
5: -8.5554590, -5.0883007, -8.5616398, -5.0880322, -2.6248231, 2.6465516
6: -12.7825661, -8.5374203, -12.7881060, -8.5355101, -3.8069592, 3.7917695
7: -9.2127581, -5.6961088, -9.2024956, -5.6787043, -3.3363495, 3.3367081
8: 9.6355495, 12.5913286, 9.6288147, 12.5982666, -2.9627171, 2.9625139
9: -7.9772363, -3.6878130, -7.9776235, -3.6958857, -3.4877806, 3.5016155

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9534201, upper bound: 1.9682911
time: 4.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9626766, upper bound: 1.9682913
time: 4.77 seconds

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

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9618951, upper bound: 1.9488608
time: 4.51 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627926, upper bound: 1.9556310
time: 4.69 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.0009174, -6.8784881, -10.9926262, -6.8969135, -3.5682669, 3.5835595
1: -9.9720984, -6.7895575, -9.9561443, -6.7805409, -3.1362290, 3.1123896
2: -4.8623590, -1.6980295, -4.8548784, -1.6982555, -3.1387691, 3.1334729
3: -1.7176828, 1.6615769, -1.7002244, 1.6284927, -3.3461757, 3.3618011
4: -14.0009222, -10.0292492, -13.9712477, -10.0436945, -3.7306767, 3.7201099
5: -8.5497885, -5.0965672, -8.5395498, -5.1094713, -2.6058989, 2.6091237
6: -12.7727537, -8.5530739, -12.7736778, -8.5507727, -3.7580519, 3.7564325
7: -9.1901245, -5.7043419, -9.1814651, -5.6894279, -3.2868967, 3.2715755
8: 9.6523981, 12.5937834, 9.6512241, 12.5806084, -2.9282103, 2.9425592
9: -7.9615707, -3.7049127, -7.9519272, -3.7113175, -3.4544668, 3.4516912

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9618955, upper bound: 1.9488608
time: 5.00 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9627930, upper bound: 1.9556311
time: 4.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0396605, -6.8301415, -10.9921064, -6.9008269, -3.5990992, 3.6463161
1: -10.0099134, -6.7672253, -9.9366131, -6.7963881, -3.1656356, 3.1225996
2: -4.8829870, -1.6793053, -4.8259621, -1.7157066, -3.1537704, 3.1466568
3: -1.7492623, 1.6770486, -1.6825564, 1.6177263, -3.3669887, 3.3596048
4: -14.0216331, -10.0042543, -13.9566708, -10.0658407, -3.7448378, 3.7470636
5: -8.5647564, -5.0829153, -8.5316038, -5.1194544, -2.6184735, 2.6211908
6: -12.7993908, -8.5303516, -12.7589903, -8.5567942, -3.7891469, 3.7731819
7: -9.2270241, -5.6739125, -9.1647472, -5.7072840, -3.3250942, 3.3058386
8: 9.6165209, 12.6077833, 9.6694489, 12.5635834, -2.9470625, 2.9383345
9: -7.9796853, -3.6856451, -7.9492912, -3.7154269, -3.4708333, 3.4709277

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9679002, upper bound: 1.9497377
time: 5.37 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687897, upper bound: 1.9564892
time: 5.29 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.0396605, -6.8301415, -11.0008259, -6.8934197, -3.6119838, 3.6564562
1: -10.0099134, -6.7672253, -9.9581585, -6.7759047, -3.1739659, 3.1365657
2: -4.8829870, -1.6793053, -4.8560114, -1.6945016, -3.1652565, 3.1598535
3: -1.7492623, 1.6770486, -1.7037055, 1.6298199, -3.3790822, 3.3807540
4: -14.0216331, -10.0042543, -13.9730225, -10.0400181, -3.7556934, 3.7491159
5: -8.5647564, -5.0829153, -8.5420904, -5.1087127, -2.6229939, 2.6255710
6: -12.7993908, -8.5303516, -12.7772369, -8.5471449, -3.7895613, 3.7823973
7: -9.2270241, -5.6739125, -9.1833286, -5.6836605, -3.3298750, 3.3053389
8: 9.6165209, 12.6077833, 9.6465311, 12.5813503, -2.9648294, 2.9612522
9: -7.9796853, -3.6856451, -7.9549861, -3.7099614, -3.4758759, 3.4766488

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9679005, upper bound: 1.9497377
time: 6.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.9687901, upper bound: 1.9564892
time: 5.06 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 32.44 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9562620, upper bound: 1.9488607
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9571791, upper bound: 1.9556310
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9562620, upper bound: 1.9544911
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9571791, upper bound: 1.9612447
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9622674, upper bound: 1.9497377
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9631764, upper bound: 1.9564894
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9622675, upper bound: 1.9553683
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9631764, upper bound: 1.9621030
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9474155, upper bound: 1.9618146
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9566756, upper bound: 1.9618149
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9474156, upper bound: 1.9674276
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9566757, upper bound: 1.9674280
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9534202, upper bound: 1.9626768
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9626766, upper bound: 1.9626770
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9534201, upper bound: 1.9682911
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9626766, upper bound: 1.9682913
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9618951, upper bound: 1.9488608
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9627926, upper bound: 1.9556310
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9618955, upper bound: 1.9488608
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9627930, upper bound: 1.9556311
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9679002, upper bound: 1.9497377
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9687897, upper bound: 1.9564892
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9679005, upper bound: 1.9497377
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 32.44
Output dim: 8, lower bound: -1.9687901, upper bound: 1.9564892
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.44
Output dim: 8, lower bound: -1.9628788, upper bound: 1.9624028
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.44
Output dim: 8, lower bound: -1.9628791, upper bound: 1.9624030
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.44
Output dim: 8, lower bound: -1.9688759, upper bound: 1.9632625
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.44
Output dim: 8, lower bound: -1.9688763, upper bound: 1.9632625
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.944035530090332
rel_dist={8: [-1.969206455838302, 1.969206699419745]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205467, upper bound: 1.6237266
time: 6.44 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237271, upper bound: 1.6237266
time: 5.00 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 11.64 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 11.64
Output dim: 8, lower bound: -1.6205467, upper bound: 1.6237266
IS_A2, status: Status.UNKNOWN, split count: 1, time: 11.64
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

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6191775
time: 6.78 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6237239
time: 5.62 seconds

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

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237240, upper bound: 1.6191780
time: 5.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237240, upper bound: 1.6237237
time: 5.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 25.89 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 25.89
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6191775
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 25.89
Output dim: 8, lower bound: -1.6205436, upper bound: 1.6237239
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 25.89
Output dim: 8, lower bound: -1.6237240, upper bound: 1.6191780
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 25.89
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

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165416, upper bound: 1.6178916
time: 6.49 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205371, upper bound: 1.6191717
time: 6.09 seconds

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

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165416, upper bound: 1.6224374
time: 5.53 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205371, upper bound: 1.6237176
time: 6.21 seconds

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

Time for backsubstitution: 14.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6197180, upper bound: 1.6178914
time: 5.98 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237174, upper bound: 1.6191717
time: 5.26 seconds

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

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6197180, upper bound: 1.6224374
time: 5.76 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237174, upper bound: 1.6237175
time: 5.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 26.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6165416, upper bound: 1.6178916
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6205371, upper bound: 1.6191717
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6165416, upper bound: 1.6224374
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6205371, upper bound: 1.6237176
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6197180, upper bound: 1.6178914
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6237174, upper bound: 1.6191717
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6197180, upper bound: 1.6224374
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 26.33
Output dim: 8, lower bound: -1.6237174, upper bound: 1.6237175

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

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6136535, upper bound: 1.6174910
time: 5.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6178885
time: 7.02 seconds

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

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6176574, upper bound: 1.6187748
time: 5.88 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6191682
time: 6.50 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.9946327, -6.8746061, -11.0134144, -6.8638268, -3.1575012, 3.2012796
1: -9.9523745, -6.8079939, -9.9641523, -6.7967162, -2.8509116, 2.8533111
2: -4.8350310, -1.7175033, -4.8387065, -1.7009654, -2.8981743, 2.8818078
3: -1.7068818, 1.6525521, -1.7172964, 1.6588089, -3.3656907, 3.3698485
4: -13.9877119, -10.0452080, -13.9985695, -10.0365429, -3.3693495, 3.3816700
5: -8.5404902, -5.1019678, -8.5493565, -5.0993338, -2.2439508, 2.2697546
6: -12.7560005, -8.5601664, -12.7644682, -8.5486641, -3.3975797, 3.3774052
7: -9.1757793, -5.7265368, -9.1878004, -5.7131462, -2.9151201, 2.9367476
8: 9.6714344, 12.5773344, 9.6537590, 12.5796003, -2.7256737, 2.7330830
9: -7.9591312, -3.7071242, -7.9664192, -3.7029841, -3.0992060, 3.1059604

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159302, upper bound: 1.6159520
time: 5.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159302, upper bound: 1.6218231
time: 5.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.0333700, -6.8262696, -11.0301380, -6.8565598, -3.2060585, 3.2651496
1: -9.9901886, -6.7856879, -9.9682961, -6.7872477, -2.8948431, 2.8782940
2: -4.8556566, -1.6987637, -4.8410387, -1.6932793, -2.9292641, 2.9080019
3: -1.7383661, 1.6680560, -1.7243011, 1.6615012, -3.3998673, 3.3923571
4: -14.0084782, -10.0202141, -14.0022507, -10.0290813, -3.3988190, 3.4115696
5: -8.5554562, -5.0883055, -8.5545578, -5.0977764, -2.2614541, 2.2883492
6: -12.7825661, -8.5374212, -12.7715368, -8.5412922, -3.4328985, 3.4068809
7: -9.2127552, -5.6961107, -9.1916866, -5.7013636, -2.9649277, 2.9704413
8: 9.6355524, 12.5913277, 9.6441488, 12.5811214, -2.7610002, 2.7588975
9: -7.9772363, -3.6878169, -7.9726624, -3.7002015, -3.1223307, 3.1343751

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6199298, upper bound: 1.6172360
time: 6.84 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6199298, upper bound: 1.6231075
time: 6.36 seconds

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

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6168300, upper bound: 1.6174911
time: 6.79 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6197147, upper bound: 1.6178881
time: 5.63 seconds

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

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6168300, upper bound: 1.6187748
time: 6.14 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6237142, upper bound: 1.6191687
time: 4.93 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0033569, -6.8672123, -11.0144339, -6.8615847, -3.1698475, 3.2140565
1: -9.9739218, -6.7875247, -9.9703951, -6.7955961, -2.8717060, 2.8743563
2: -4.8650694, -1.6962702, -4.8397064, -1.6936448, -2.9241214, 2.8993187
3: -1.7279301, 1.6645770, -1.7199252, 1.6617935, -3.3897235, 3.3845022
4: -14.0040331, -10.0193872, -14.0045004, -10.0357590, -3.3860626, 3.4009628
5: -8.5509996, -5.0912352, -8.5523014, -5.0984902, -2.2535381, 2.2801890
6: -12.7742004, -8.5505133, -12.7659626, -8.5453348, -3.4190245, 3.3853478
7: -9.1943130, -5.7029176, -9.1944609, -5.7123141, -2.9296474, 2.9669309
8: 9.6484909, 12.5951052, 9.6472712, 12.5801306, -2.7497082, 2.7590005
9: -7.9648361, -3.7016664, -7.9670706, -3.7020128, -3.1065087, 3.1133299

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6191066, upper bound: 1.6159518
time: 6.22 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6191066, upper bound: 1.6218232
time: 7.67 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.0420942, -6.8188524, -11.0311584, -6.8543153, -3.2183747, 3.2753644
1: -10.0117273, -6.7651949, -9.9745388, -6.7861261, -2.9158311, 2.8993819
2: -4.8857145, -1.6775440, -4.8420401, -1.6859652, -2.9494548, 2.9254818
3: -1.7594852, 1.6800313, -1.7269373, 1.6644850, -3.4239702, 3.4069686
4: -14.0247602, -9.9943972, -14.0081778, -10.0282974, -3.4156027, 3.4312243
5: -8.5659580, -5.0775824, -8.5574989, -5.0969300, -2.2710690, 2.2961023
6: -12.8008480, -8.5277987, -12.7730236, -8.5379639, -3.4543619, 3.4148173
7: -9.2313032, -5.6724882, -9.1983500, -5.7005286, -2.9795303, 3.0006342
8: 9.6126232, 12.6091051, 9.6376677, 12.5816498, -2.7850528, 2.7818291
9: -7.9829483, -3.6823862, -7.9733095, -3.6992276, -3.1296554, 3.1417427

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6231075, upper bound: 1.6172357
time: 5.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6231075, upper bound: 1.6231074
time: 6.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 26.92 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6136535, upper bound: 1.6174910
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6165383, upper bound: 1.6178885
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6176574, upper bound: 1.6187748
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6205338, upper bound: 1.6191682
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6159302, upper bound: 1.6159520
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6159302, upper bound: 1.6218231
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6199298, upper bound: 1.6172360
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6199298, upper bound: 1.6231075
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6168300, upper bound: 1.6174911
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6197147, upper bound: 1.6178881
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6168300, upper bound: 1.6187748
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6237142, upper bound: 1.6191687
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6191066, upper bound: 1.6159518
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6191066, upper bound: 1.6218232
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6231075, upper bound: 1.6172357
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.92
Output dim: 8, lower bound: -1.6231075, upper bound: 1.6231074

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

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6103349, upper bound: 1.6174911
time: 5.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6103349, upper bound: 1.6174918
time: 7.28 seconds

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

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6132194, upper bound: 1.6178886
time: 5.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6132194, upper bound: 1.6178883
time: 5.25 seconds

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

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6143393, upper bound: 1.6187748
time: 5.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6143392, upper bound: 1.6187754
time: 7.52 seconds

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

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6172138, upper bound: 1.6191683
time: 8.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6172138, upper bound: 1.6191682
time: 5.58 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.9941978, -6.8764477, -11.0123920, -6.8681178, -3.1503439, 3.1971836
1: -9.9517155, -6.8108201, -9.9626141, -6.8032990, -2.8425827, 2.8482761
2: -4.8325319, -1.7179941, -4.8328857, -1.7021101, -2.8945141, 2.8753386
3: -1.7028091, 1.6507335, -1.7078171, 1.6545671, -3.3573761, 3.3585505
4: -13.9857540, -10.0508165, -13.9939957, -10.0495815, -3.3546314, 3.3729591
5: -8.5399618, -5.1063590, -8.5481186, -5.1095643, -2.2332859, 2.2643275
6: -12.7555389, -8.5616360, -12.7633839, -8.5520935, -3.3930831, 3.3752384
7: -9.1743164, -5.7271166, -9.1843700, -5.7144966, -2.9119911, 2.9310551
8: 9.6746197, 12.5767374, 9.6611710, 12.5781918, -2.7212200, 2.7250965
9: -7.9581089, -3.7074809, -7.9640360, -3.7038188, -3.0949535, 3.1023245

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6155289, upper bound: 1.6130629
time: 5.38 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159272, upper bound: 1.6159482
time: 6.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.9946222, -6.8746119, -11.0207672, -6.8626051, -3.1530676, 3.2040200
1: -9.9523716, -6.8080387, -9.9859772, -6.7958951, -2.8480482, 2.8681931
2: -4.8350220, -1.7175536, -4.8425260, -1.6910260, -2.9081507, 2.8833237
3: -1.7068119, 1.6525447, -1.7199860, 1.6761580, -3.3829699, 3.3725307
4: -13.9877014, -10.0452385, -14.0179377, -10.0353785, -3.3679390, 3.3879447
5: -8.5404510, -5.1019850, -8.5664558, -5.0973167, -2.2417140, 2.2737377
6: -12.7550011, -8.5601721, -12.7645674, -8.5457821, -3.4088392, 3.3777084
7: -9.1757736, -5.7298913, -9.1904478, -5.7158256, -2.9127207, 2.9516206
8: 9.6715498, 12.5773335, 9.6514845, 12.5920334, -2.7384033, 2.7335877
9: -7.9569850, -3.7071431, -7.9648666, -3.7038648, -3.0923343, 3.1146727

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6155289, upper bound: 1.6189339
time: 6.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6159272, upper bound: 1.6218199
time: 4.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.0329313, -6.8281240, -11.0291185, -6.8608589, -3.1988974, 3.2587543
1: -9.9895277, -6.7885151, -9.9667597, -6.7938294, -2.8864965, 2.8732562
2: -4.8531513, -1.6992545, -4.8352156, -1.6944240, -2.9255972, 2.9015341
3: -1.7342827, 1.6662579, -1.7148399, 1.6572634, -3.3915462, 3.3810978
4: -14.0065136, -10.0258226, -13.9976654, -10.0421000, -3.3840799, 3.4028602
5: -8.5549307, -5.0927000, -8.5533209, -5.1080070, -2.2507906, 2.2818480
6: -12.7820988, -8.5388775, -12.7704544, -8.5447121, -3.4284339, 3.4047236
7: -9.2112627, -5.6966848, -9.1882429, -5.7027102, -2.9617701, 2.9647326
8: 9.6387348, 12.5907269, 9.6515617, 12.5797119, -2.7565508, 2.7509062
9: -7.9762220, -3.6881754, -7.9702821, -3.7010338, -3.1180897, 3.1307168

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6195300, upper bound: 1.6143468
time: 5.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6199268, upper bound: 1.6172324
time: 6.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -11.0333576, -6.8262773, -11.0374889, -6.8553386, -3.2016215, 3.2653289
1: -9.9901829, -6.7857342, -9.9901257, -6.7864265, -2.8924818, 2.8931808
2: -4.8556480, -1.6988146, -4.8448639, -1.6833417, -2.9334664, 2.9095140
3: -1.7382977, 1.6680489, -1.7270288, 1.6788428, -3.4171405, 3.3950777
4: -14.0084696, -10.0202446, -14.0216160, -10.0278988, -3.3974066, 3.4182072
5: -8.5554161, -5.0883236, -8.5716581, -5.0957594, -2.2592154, 2.2896531
6: -12.7815609, -8.5374279, -12.7716408, -8.5384102, -3.4441557, 3.4073300
7: -9.2127504, -5.6994481, -9.1943331, -5.7040386, -2.9625359, 2.9853201
8: 9.6356678, 12.5913239, 9.6418762, 12.5935555, -2.7737298, 2.7594035
9: -7.9750662, -3.6878383, -7.9711051, -3.7010794, -3.1154594, 3.1430836

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6195300, upper bound: 1.6202185
time: 5.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6199268, upper bound: 1.6231039
time: 5.91 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.9876556, -6.8903427, -10.9535809, -6.9188461, -3.1304502, 3.1403894
1: -9.9648380, -6.8032255, -9.9224520, -6.8297353, -2.8269305, 2.8050835
2: -4.8535142, -1.7133890, -4.8052120, -1.7365364, -2.8744826, 2.8499584
3: -1.6884975, 1.6552577, -1.6336935, 1.5932496, -3.2817471, 3.2889512
4: -13.9831514, -10.0391798, -13.9358549, -10.0778198, -3.3273058, 3.3064332
5: -8.5397940, -5.1032395, -8.5133762, -5.1338167, -2.2081156, 2.2299211
6: -12.7586288, -8.5571594, -12.7285786, -8.5683270, -3.3677778, 3.3474755
7: -9.1763563, -5.7077780, -9.1521854, -5.7255177, -2.8997345, 2.8883147
8: 9.6619215, 12.5800438, 9.6836052, 12.5372810, -2.6815009, 2.7083654
9: -7.9534116, -3.7217853, -7.9197955, -3.7464805, -3.0472350, 3.0198350

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6135114, upper bound: 1.6174912
time: 5.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6135114, upper bound: 1.6174917
time: 10.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.9990349, -6.8870964, -10.9775543, -6.9032497, -3.1614394, 3.1622992
1: -9.9706917, -6.7911148, -9.9459753, -6.8034434, -2.8594179, 2.8448505
2: -4.8602948, -1.6993915, -4.8257976, -1.7075633, -2.9080505, 2.8854551
3: -1.7098451, 1.6592696, -1.6812291, 1.6214826, -3.3313277, 3.3404987
4: -13.9985170, -10.0367775, -13.9658356, -10.0716324, -3.3382502, 3.3773756
5: -8.5488558, -5.1006384, -8.5327396, -5.1191831, -2.2492890, 2.2508662
6: -12.7716484, -8.5550203, -12.7551765, -8.5569887, -3.3936357, 3.3683381
7: -9.1869431, -5.7054224, -9.1753302, -5.7172637, -2.9152927, 2.9410410
8: 9.6553822, 12.5927610, 9.6650124, 12.5632019, -2.7192025, 2.7387576
9: -7.9590569, -3.7073929, -7.9444494, -3.7161324, -3.0849395, 3.0808043

Time for backsubstitution: 14.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6163961, upper bound: 1.6178888
time: 5.52 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6163961, upper bound: 1.6178887
time: 8.67 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0263996, -6.8420143, -10.9703312, -6.9115906, -3.1789722, 3.2163358
1: -10.0026665, -6.7808862, -9.9266138, -6.8202615, -2.8742499, 2.8301170
2: -4.8741207, -1.6946716, -4.8075361, -1.7288677, -2.8997588, 2.8761606
3: -1.7200902, 1.6707693, -1.6407287, 1.5959669, -3.3160572, 3.3114982
4: -14.0038385, -10.0141869, -13.9395084, -10.0703201, -3.3567677, 3.3363543
5: -8.5547676, -5.0895863, -8.5185823, -5.1322680, -2.2255931, 2.2493684
6: -12.7852440, -8.5344143, -12.7356243, -8.5609360, -3.4032125, 3.3769493
7: -9.2131805, -5.6773295, -9.1560259, -5.7137313, -2.9492750, 2.9219508
8: 9.6259956, 12.5940418, 9.6739950, 12.5387993, -2.7169437, 2.7318110
9: -7.9715405, -3.7025323, -7.9260464, -3.7436850, -3.0703974, 3.0481844

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6175153, upper bound: 1.6187752
time: 5.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.6175154, upper bound: 1.6187754
time: 5.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 31.72 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6103349, upper bound: 1.6174911
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6103349, upper bound: 1.6174918
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6132194, upper bound: 1.6178886
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6132194, upper bound: 1.6178883
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6143393, upper bound: 1.6187748
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6143392, upper bound: 1.6187754
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6172138, upper bound: 1.6191683
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6172138, upper bound: 1.6191682
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6155289, upper bound: 1.6130629
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6159272, upper bound: 1.6159482
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6155289, upper bound: 1.6189339
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6159272, upper bound: 1.6218199
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6195300, upper bound: 1.6143468
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6199268, upper bound: 1.6172324
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6195300, upper bound: 1.6202185
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6199268, upper bound: 1.6231039
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6135114, upper bound: 1.6174912
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6135114, upper bound: 1.6174917
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6163961, upper bound: 1.6178888
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6163961, upper bound: 1.6178887
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6175153, upper bound: 1.6187752
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 31.72
Output dim: 8, lower bound: -1.6175154, upper bound: 1.6187754
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 8, lower bound: -1.6237142, upper bound: 1.6191687
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 8, lower bound: -1.6191066, upper bound: 1.6159518
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 8, lower bound: -1.6191066, upper bound: 1.6218232
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 8, lower bound: -1.6231075, upper bound: 1.6172357
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.72
Output dim: 8, lower bound: -1.6231075, upper bound: 1.6231074
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.7666475772857666
rel_dist={8: [-1.6237368538932522, 1.6237361751053658]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5845
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5845

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930243, upper bound: 1.4952811
time: 11.04 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952818, upper bound: 1.4952822
time: 7.80 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 19.05 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 19.05
Output dim: 8, lower bound: -1.4930243, upper bound: 1.4952811
IS_A2, status: Status.UNKNOWN, split count: 1, time: 19.05
Output dim: 8, lower bound: -1.4952818, upper bound: 1.4952822

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -11.0289745, -6.8591261, -11.0299339, -6.8570595, -3.1110601, 3.1093917
1: -9.9610510, -6.7885365, -9.9668102, -6.7874885, -2.7875094, 2.7918425
2: -4.8398800, -1.7017744, -4.8408098, -1.6950119, -2.8390427, 2.8322558
3: -1.7212706, 1.6580515, -1.7237175, 1.6608000, -3.3820705, 3.3817689
4: -13.9953804, -10.0299664, -14.0008450, -10.0292339, -3.2779102, 3.2829885
5: -8.5511646, -5.0987401, -8.5538702, -5.0979548, -2.1599288, 2.1619568
6: -12.7698078, -8.5451317, -12.7711935, -8.5420647, -3.2905378, 3.2889876
7: -9.1839590, -5.7023129, -9.1901073, -5.7015324, -2.8337574, 2.8390956
8: 9.6516819, 12.5805035, 9.6456776, 12.5809994, -2.6667399, 2.6733277
9: -7.9719172, -3.7013087, -7.9725299, -3.7004180, -2.9964943, 2.9957128

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930220, upper bound: 1.4918504
time: 7.55 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930219, upper bound: 1.4952784
time: 11.00 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -11.0377007, -6.8517303, -11.0312014, -6.8542767, -3.1243005, 3.1244564
1: -9.9825935, -6.7680578, -9.9745502, -6.7860970, -2.8081865, 2.8194935
2: -4.8699226, -1.6805701, -4.8420515, -1.6859413, -2.8718596, 2.8484454
3: -1.7423849, 1.6700652, -1.7269831, 1.6645001, -3.4068851, 3.3970482
4: -14.0116749, -10.0041485, -14.0081930, -10.0282612, -3.2936316, 3.3167553
5: -8.5616531, -5.0880113, -8.5575171, -5.0969095, -2.1693201, 2.1761422
6: -12.7881174, -8.5354910, -12.7730370, -8.5379372, -3.3128376, 3.2967768
7: -9.2025137, -5.6786847, -9.1983671, -5.7004967, -2.8473597, 2.8710032
8: 9.6287947, 12.5982733, 9.6376410, 12.5816555, -2.6898413, 2.7011070
9: -7.9776421, -3.6958699, -7.9733362, -3.6992116, -3.0041790, 3.0030961

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5762
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5762

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4918480
time: 10.00 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4952787
time: 11.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 35.92 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 35.92
Output dim: 8, lower bound: -1.4930220, upper bound: 1.4918504
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 35.92
Output dim: 8, lower bound: -1.4930219, upper bound: 1.4952784
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 35.92
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4918480
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 35.92
Output dim: 8, lower bound: -1.4952794, upper bound: 1.4952787

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -11.0238571, -6.8827548, -10.9930849, -6.8987532, -3.0657387, 3.0520239
1: -9.9572439, -6.7927899, -9.9423857, -6.7953253, -2.7727747, 2.7607164
2: -4.8341970, -1.7054816, -4.8268929, -1.7089313, -2.8188925, 2.8131952
3: -1.6998494, 1.6517212, -1.6850274, 1.6204882, -3.3203375, 3.3367486
4: -13.9887581, -10.0505257, -13.9621506, -10.0650978, -3.2369556, 3.2270446
5: -8.5486145, -5.1099234, -8.5343113, -5.1186700, -2.1362739, 2.1304939
6: -12.7667665, -8.5504627, -12.7603817, -8.5537138, -3.2715688, 3.2686586
7: -9.1751575, -5.7052865, -9.1709089, -5.7064896, -2.8140078, 2.8157616
8: 9.6598654, 12.5777187, 9.6634302, 12.5640783, -2.6402063, 2.6507885
9: -7.9650517, -3.7081127, -7.9499078, -3.7145298, -2.9749045, 2.9651313

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897180, upper bound: 1.4906140
time: 7.71 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930156, upper bound: 1.4918423
time: 8.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -11.0289717, -6.8591423, -11.0299263, -6.8570890, -3.0749030, 3.1093650
1: -9.9610481, -6.7885408, -9.9668045, -6.7874947, -2.7851267, 2.7918291
2: -4.8398738, -1.7017782, -4.8408012, -1.6950197, -2.8390284, 2.8277097
3: -1.7212491, 1.6580458, -1.7236814, 1.6607888, -3.3820379, 3.3817272
4: -13.9953718, -10.0299778, -14.0008316, -10.0292549, -3.2684126, 3.2829609
5: -8.5511627, -5.0987520, -8.5538645, -5.0979757, -2.1399317, 2.1619420
6: -12.7698030, -8.5451365, -12.7711906, -8.5420723, -3.2965393, 3.2778230
7: -9.1839495, -5.7023163, -9.1900921, -5.7015362, -2.8215790, 2.8486061
8: 9.6516857, 12.5805016, 9.6456871, 12.5809956, -2.6667275, 2.6648302
9: -7.9719100, -3.7013152, -7.9725204, -3.7004280, -2.9952974, 2.9988704

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897179, upper bound: 1.4940420
time: 15.07 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930155, upper bound: 1.4952724
time: 6.38 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -11.0325794, -6.8753543, -10.9943485, -6.8959670, -3.0789747, 3.0670853
1: -9.9787693, -6.7723098, -9.9501400, -6.7939301, -2.7934303, 2.7871864
2: -4.8642459, -1.6842812, -4.8281298, -1.6998584, -2.8510132, 2.8293829
3: -1.7209897, 1.6637623, -1.6883104, 1.6242009, -3.3451905, 3.3520727
4: -14.0050774, -10.0247059, -13.9695110, -10.0641222, -3.2526417, 3.2607956
5: -8.5591030, -5.0991921, -8.5379543, -5.1176276, -2.1456513, 2.1446905
6: -12.7850676, -8.5408173, -12.7622185, -8.5495815, -3.2938437, 3.2764411
7: -9.1937227, -5.6816602, -9.1791821, -5.7054539, -2.8275728, 2.8476415
8: 9.6369715, 12.5954819, 9.6553841, 12.5647354, -2.6633091, 2.6785669
9: -7.9707479, -3.7026713, -7.9507089, -3.7133224, -2.9825583, 2.9725137

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919754, upper bound: 1.4906140
time: 7.15 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952730, upper bound: 1.4918421
time: 8.67 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -11.0376987, -6.8517475, -11.0311956, -6.8543077, -3.0881433, 3.1244278
1: -9.9825897, -6.7680626, -9.9745455, -6.7861028, -2.8058023, 2.8131421
2: -4.8699174, -1.6805743, -4.8420429, -1.6859481, -2.8679562, 2.8439016
3: -1.7423668, 1.6700581, -1.7269540, 1.6644895, -3.4068563, 3.3970122
4: -14.0116692, -10.0041599, -14.0081835, -10.0282822, -3.2841330, 3.3026552
5: -8.5616484, -5.0880227, -8.5575123, -5.0969291, -2.1493230, 2.1730077
6: -12.7881193, -8.5354958, -12.7730360, -8.5379467, -3.3188229, 3.2856317
7: -9.2025051, -5.6786857, -9.1983538, -5.7005000, -2.8352127, 2.8805113
8: 9.6288004, 12.5982695, 9.6376524, 12.5816507, -2.6898274, 2.6926105
9: -7.9776363, -3.6958761, -7.9733248, -3.6992211, -3.0029831, 3.0062528

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4627

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919754, upper bound: 1.4940421
time: 8.10 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952730, upper bound: 1.4952723
time: 6.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.66 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4897180, upper bound: 1.4906140
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4930156, upper bound: 1.4918423
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4897179, upper bound: 1.4940420
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4930155, upper bound: 1.4952724
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4919754, upper bound: 1.4906140
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4952730, upper bound: 1.4918421
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4919754, upper bound: 1.4940421
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 8, lower bound: -1.4952730, upper bound: 1.4952723

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.9895029, -6.8982143, -10.9727230, -6.9076414, -3.0109091, 3.0063324
1: -9.9485550, -6.8122468, -9.9373140, -6.8068409, -2.7508221, 2.7345314
2: -4.8293648, -1.7211999, -4.8240614, -1.7182562, -2.7961040, 2.7876048
3: -1.6854138, 1.6462166, -1.6764765, 1.6171958, -3.3026097, 3.3226931
4: -13.9811306, -10.0658398, -13.9576855, -10.0741997, -3.2143250, 3.2019625
5: -8.5379391, -5.1131368, -8.5279922, -5.1205549, -2.1220164, 2.1195865
6: -12.7529879, -8.5655193, -12.7519388, -8.5626869, -3.2472882, 3.2441306
7: -9.1670380, -5.7295017, -9.1662178, -5.7208118, -2.7879305, 2.7832270
8: 9.6796188, 12.5745602, 9.6751146, 12.5622234, -2.6160817, 2.6324043
9: -7.9522772, -3.7139189, -7.9423208, -3.7179391, -2.9570007, 2.9499550

Time for backsubstitution: 14.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4872950, upper bound: 1.4900398
time: 10.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4897156, upper bound: 1.4906100
time: 7.48 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -11.0282440, -6.8498998, -10.9930391, -6.8987646, -3.0615396, 3.0895076
1: -9.9863911, -6.7899356, -9.9423771, -6.7953525, -2.7989488, 2.7600074
2: -4.8499508, -1.7024696, -4.8268886, -1.7089542, -2.8290601, 2.8140125
3: -1.7169554, 1.6617647, -1.6850064, 1.6204828, -3.3374381, 3.3467712
4: -14.0018616, -10.0408497, -13.9621420, -10.0651169, -3.2455864, 3.2323847
5: -8.5529165, -5.0994802, -8.5342960, -5.1186738, -2.1397614, 2.1402869
6: -12.7795258, -8.5427513, -12.7603683, -8.5537357, -3.2842665, 3.2750921
7: -9.2038507, -5.6990781, -9.1709042, -5.7065191, -2.8401375, 2.8170276
8: 9.6437197, 12.5885458, 9.6634541, 12.5640764, -2.6513591, 2.6603293
9: -7.9703860, -3.6946390, -7.9498916, -3.7145352, -2.9808693, 2.9797230

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4905901, upper bound: 1.4912665
time: 7.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4930132, upper bound: 1.4918401
time: 23.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.9946327, -6.8746090, -11.0096006, -6.8659782, -3.0200653, 3.0636511
1: -9.9523735, -6.8079939, -9.9617519, -6.7990084, -2.7631712, 2.7656531
2: -4.8350296, -1.7175051, -4.8379602, -1.7043579, -2.8162560, 2.8021479
3: -1.7068772, 1.6525511, -1.7151649, 1.6575224, -3.3643997, 3.3614306
4: -13.9877090, -10.0452089, -13.9963512, -10.0383139, -3.2457790, 3.2578564
5: -8.5404892, -5.1019697, -8.5475407, -5.0998688, -2.1256609, 2.1510205
6: -12.7560024, -8.5601673, -12.7626991, -8.5510254, -3.2723632, 3.2533207
7: -9.1757784, -5.7265363, -9.1853456, -5.7158642, -2.7954245, 2.8160014
8: 9.6714382, 12.5773344, 9.6573715, 12.5791407, -2.6426182, 2.6464570
9: -7.9591284, -3.7071240, -7.9649353, -3.7038229, -2.9773684, 2.9836502

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891316, upper bound: 1.4890443
time: 16.98 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891316, upper bound: 1.4934582
time: 9.18 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.0333672, -6.8262739, -11.0298843, -6.8570991, -3.0707340, 3.1294222
1: -9.9901848, -6.7856874, -9.9667950, -6.7875247, -2.8075829, 2.7911158
2: -4.8556547, -1.6987646, -4.8407969, -1.6950406, -2.8492775, 2.8285179
3: -1.7383623, 1.6680560, -1.7236624, 1.6607838, -3.3991461, 3.3905869
4: -14.0084772, -10.0202179, -14.0008259, -10.0292711, -3.2771082, 3.2882996
5: -8.5554552, -5.0883074, -8.5538492, -5.0979781, -2.1433878, 2.1698830
6: -12.7825642, -8.5374241, -12.7711744, -8.5420923, -3.3092608, 3.2844033
7: -9.2127523, -5.6961107, -9.1900854, -5.7015667, -2.8479853, 2.8498917
8: 9.6355553, 12.5913267, 9.6457119, 12.5809917, -2.6778579, 2.6743832
9: -7.9772334, -3.6878188, -7.9725003, -3.7004342, -3.0012236, 3.0134869

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924316, upper bound: 1.4902770
time: 8.16 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924316, upper bound: 1.4946892
time: 6.88 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -10.9982204, -6.8908157, -10.9739895, -6.9048624, -3.0241899, 3.0212164
1: -9.9700851, -6.7917767, -9.9450684, -6.8054485, -2.7714500, 2.7610512
2: -4.8594079, -1.6999687, -4.8252969, -1.7091743, -2.8265901, 2.8037839
3: -1.7064650, 1.6582699, -1.6797450, 1.6209102, -3.3273752, 3.3380148
4: -13.9974775, -10.0400219, -13.9650517, -10.0732241, -3.2299824, 3.2357044
5: -8.5484495, -5.1023993, -8.5316353, -5.1195145, -2.1313620, 2.1337876
6: -12.7711678, -8.5558596, -12.7537842, -8.5585527, -3.2695217, 3.2519236
7: -9.1855764, -5.7058883, -9.1744900, -5.7197790, -2.8014698, 2.8150887
8: 9.6566696, 12.5923233, 9.6670618, 12.5628805, -2.6391349, 2.6601872
9: -7.9579706, -3.7084587, -7.9431238, -3.7167244, -2.9646516, 2.9573307

Time for backsubstitution: 14.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4895523, upper bound: 1.4900386
time: 6.04 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4919729, upper bound: 1.4906099
time: 7.70 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -11.0369682, -6.8424811, -10.9943066, -6.8959799, -3.0747824, 3.0995741
1: -10.0079126, -6.7694421, -9.9501324, -6.7939591, -2.8198223, 2.7865593
2: -4.8800125, -1.6812503, -4.8281255, -1.6998799, -2.8536668, 2.8301640
3: -1.7380745, 1.6737665, -1.6882904, 1.6241956, -3.3622701, 3.3620567
4: -14.0181665, -10.0150280, -13.9695053, -10.0641403, -3.2613077, 3.2661219
5: -8.5634232, -5.0887518, -8.5379372, -5.1176319, -2.1491356, 2.1544814
6: -12.7977962, -8.5331240, -12.7622061, -8.5496044, -3.3065171, 3.2828684
7: -9.2223959, -5.6754599, -9.1791744, -5.7054868, -2.8537550, 2.8488984
8: 9.6207848, 12.6063213, 9.6554079, 12.5647287, -2.6744289, 2.6872931
9: -7.9760885, -3.6892056, -7.9506950, -3.7133317, -2.9885321, 2.9870954

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6253

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4928463, upper bound: 1.4912663
time: 23.06 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4952706, upper bound: 1.4918399
time: 8.46 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.0033541, -6.8672156, -11.0108652, -6.8631997, -3.0333486, 3.0750561
1: -9.9739208, -6.7875261, -9.9694901, -6.7976184, -2.7838278, 2.7870021
2: -4.8650675, -1.6962695, -4.8392019, -1.6952791, -2.8435287, 2.8183355
3: -1.7279251, 1.6645761, -1.7184296, 1.6612220, -3.3891470, 3.3751225
4: -14.0040321, -10.0193901, -14.0037003, -10.0373392, -3.2614603, 3.2772312
5: -8.5509977, -5.0912371, -8.5511932, -5.0988240, -2.1350222, 2.1612244
6: -12.7741995, -8.5505142, -12.7645512, -8.5468960, -3.2946033, 3.2611451
7: -9.1943102, -5.7029171, -9.1936102, -5.7148294, -2.8090353, 2.8478885
8: 9.6484938, 12.5951052, 9.6493235, 12.5798016, -2.6656699, 2.6742415
9: -7.9648352, -3.7016675, -7.9657421, -3.7026174, -2.9850335, 2.9910250

Time for backsubstitution: 14.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913911, upper bound: 1.4890432
time: 6.18 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4913911, upper bound: 1.4934575
time: 6.55 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -11.0420952, -6.8188581, -11.0311489, -6.8543200, -3.0839787, 3.1394882
1: -10.0117264, -6.7651954, -9.9745378, -6.7861304, -2.8284216, 2.8125153
2: -4.8857136, -1.6775451, -4.8420386, -1.6859686, -2.8706446, 2.8446751
3: -1.7594800, 1.6800288, -1.7269332, 1.6644828, -3.4239628, 3.4042473
4: -14.0247612, -9.9943991, -14.0081749, -10.0282993, -3.2928629, 3.3079972
5: -8.5659580, -5.0775833, -8.5574932, -5.0969324, -2.1527777, 2.1783035
6: -12.8008490, -8.5277996, -12.7730207, -8.5379667, -3.3315201, 3.2922111
7: -9.2313032, -5.6724887, -9.1983471, -5.7005329, -2.8616705, 2.8817854
8: 9.6126242, 12.6091061, 9.6376762, 12.5816498, -2.7009263, 2.6980221
9: -7.9829473, -3.6823890, -7.9733081, -3.6992307, -3.0089097, 3.0208583

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 6253
type: B, layer: 1, pos: 5845
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 5746
type: B, layer: 1, pos: 4630
type: B, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 848

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4902788
time: 9.62 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4946914
time: 7.48 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.66 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4872950, upper bound: 1.4900398
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4897156, upper bound: 1.4906100
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4905901, upper bound: 1.4912665
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4930132, upper bound: 1.4918401
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4891316, upper bound: 1.4890443
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4891316, upper bound: 1.4934582
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4924316, upper bound: 1.4902770
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4924316, upper bound: 1.4946892
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4895523, upper bound: 1.4900386
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4919729, upper bound: 1.4906099
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4928463, upper bound: 1.4912663
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4952706, upper bound: 1.4918399
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4913911, upper bound: 1.4890432
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4913911, upper bound: 1.4934575
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4902788
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.66
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4946914

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.9760513, -6.9020772, -10.9487391, -6.9232464, -2.9794855, 2.9834065
1: -9.9415369, -6.8265676, -9.9137678, -6.8331585, -2.7167654, 2.6931281
2: -4.8214636, -1.7376864, -4.8034716, -1.7472513, -2.7611151, 2.7498507
3: -1.6601441, 1.6414323, -1.6289282, 1.5889490, -3.2402306, 3.2703605
4: -13.9630947, -10.0686893, -13.9277086, -10.0803814, -3.1860209, 3.1469021
5: -8.5272312, -5.1162262, -8.5086250, -5.1351910, -2.0786238, 2.0953469
6: -12.7376118, -8.5680599, -12.7253494, -8.5740337, -3.2190628, 3.2147493
7: -9.1546011, -5.7322884, -9.1430597, -5.7290730, -2.7643032, 2.7375779
8: 9.6873550, 12.5595484, 9.6937180, 12.5362921, -2.5718575, 2.5934815
9: -7.9455090, -3.7309415, -7.9176712, -3.7483044, -2.9162617, 2.8853550

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4848586, upper bound: 1.4900393
time: 9.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4848586, upper bound: 1.4900387
time: 5.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.9894981, -6.8982158, -10.9727144, -6.9076462, -3.0126886, 3.0059013
1: -9.9485550, -6.8122559, -9.9373112, -6.8068576, -2.7501926, 2.7371387
2: -4.8293619, -1.7212095, -4.8240557, -1.7182763, -2.7960854, 2.7877998
3: -1.6854055, 1.6462154, -1.6764607, 1.6171937, -3.3025992, 3.3201895
4: -13.9811268, -10.0658398, -13.9576769, -10.0742025, -3.1977358, 3.2256308
5: -8.5379362, -5.1131382, -8.5279846, -5.1205559, -2.1220107, 2.1167088
6: -12.7529821, -8.5655184, -12.7519321, -8.5626907, -3.2472811, 3.2354889
7: -9.1670351, -5.7295012, -9.1662130, -5.7208128, -2.7806320, 2.7897477
8: 9.6796207, 12.5745535, 9.6751194, 12.5622158, -2.6097574, 2.6260562
9: -7.9522748, -3.7139268, -7.9423208, -3.7179470, -2.9548836, 2.9499440

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4872792, upper bound: 1.4906092
time: 14.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4872792, upper bound: 1.4906091
time: 5.18 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0147877, -6.8537722, -10.9690599, -6.9143767, -3.0301380, 3.0667820
1: -9.9793758, -6.8042536, -9.9188337, -6.8216696, -2.7648458, 2.7186260
2: -4.8420439, -1.7189543, -4.8062973, -1.7379506, -2.7940464, 2.7762604
3: -1.6916697, 1.6570079, -1.6374593, 1.5922415, -3.2839112, 3.2944672
4: -13.9838152, -10.0437021, -13.9321547, -10.0712910, -3.2172680, 3.1773729
5: -8.5422115, -5.1025639, -8.5149393, -5.1333065, -2.0962996, 2.1160498
6: -12.7641363, -8.5452757, -12.7338009, -8.5650768, -3.2560616, 3.2456961
7: -9.1913891, -5.7018390, -9.1477423, -5.7147722, -2.8165016, 2.7713928
8: 9.6514072, 12.5735321, 9.6820621, 12.5381413, -2.6071987, 2.6214318
9: -7.9636278, -3.7116570, -7.9252453, -3.7448978, -2.9401445, 2.9151177

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881537, upper bound: 1.4912670
time: 6.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4881537, upper bound: 1.4912663
time: 5.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -11.0282354, -6.8499012, -10.9930334, -6.8987694, -3.0633197, 3.0869212
1: -9.9863873, -6.7899466, -9.9423733, -6.7953701, -2.7944894, 2.7626152
2: -4.8499479, -1.7024783, -4.8268838, -1.7089731, -2.8290429, 2.8142080
3: -1.7169466, 1.6617622, -1.6849909, 1.6204791, -3.3374257, 3.3467531
4: -14.0018578, -10.0408497, -13.9621344, -10.0651150, -3.2290239, 3.2560778
5: -8.5529146, -5.0994806, -8.5342913, -5.1186757, -2.1397552, 2.1374092
6: -12.7795210, -8.5427532, -12.7603579, -8.5537367, -3.2842579, 3.2664509
7: -9.2038460, -5.6990824, -9.1708975, -5.7065220, -2.8328390, 2.8236051
8: 9.6437235, 12.5885429, 9.6634579, 12.5640678, -2.6450977, 2.6539874
9: -7.9703817, -3.6946428, -7.9498892, -3.7145493, -2.9787493, 2.9797111

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 848

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5762

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4905768, upper bound: 1.4918403
time: 6.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4905768, upper bound: 1.4918391
time: 8.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.9940739, -6.8769622, -11.0085802, -6.8702693, -3.0127602, 3.0578232
1: -9.9515324, -6.8116045, -9.9602127, -6.8055906, -2.7546282, 2.7597160
2: -4.8318377, -1.7181313, -4.8321419, -1.7055027, -2.8118806, 2.7955518
3: -1.7016733, 1.6502237, -1.7056849, 1.6532779, -3.3549511, 3.3502841
4: -13.9852076, -10.0523739, -13.9917746, -10.0513611, -3.2307081, 3.2475424
5: -8.5398121, -5.1075821, -8.5463028, -5.1100988, -2.1148710, 2.1443758
6: -12.7554102, -8.5620461, -12.7616177, -8.5544548, -3.2672653, 3.2500648
7: -9.1739063, -5.7272782, -9.1819229, -5.7172160, -2.7919626, 2.8093362
8: 9.6755056, 12.5765686, 9.6647854, 12.5777359, -2.6372747, 2.6383195
9: -7.9578223, -3.7075813, -7.9625502, -3.7046580, -2.9728527, 2.9796181

Time for backsubstitution: 14.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4885617, upper bound: 1.4866129
time: 5.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891294, upper bound: 1.4890414
time: 6.25 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -10.9946213, -6.8746176, -11.0169487, -6.8647585, -3.0148163, 3.0651660
1: -9.9523706, -6.8080487, -9.9835758, -6.7981873, -2.7595491, 2.7794521
2: -4.8350191, -1.7175623, -4.8417788, -1.6944191, -2.8256741, 2.8032026
3: -1.7067955, 1.6525418, -1.7178564, 1.6748738, -3.3742642, 3.3639975
4: -13.9876995, -10.0452490, -14.0157213, -10.0371552, -3.2439079, 3.2625546
5: -8.5404434, -5.1019878, -8.5646439, -5.0978556, -2.1224966, 2.1541059
6: -12.7548151, -8.5601730, -12.7628002, -8.5481396, -3.2832704, 3.2537847
7: -9.1757679, -5.7298951, -9.1879969, -5.7185431, -2.7930193, 2.8308120
8: 9.6715708, 12.5773315, 9.6550989, 12.5915766, -2.6553245, 2.6466289
9: -7.9565940, -3.7071476, -7.9633794, -3.7047012, -2.9697351, 2.9918194

Time for backsubstitution: 14.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4885619, upper bound: 1.4910286
time: 5.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4891294, upper bound: 1.4934560
time: 9.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.0328083, -6.8286428, -11.0288668, -6.8613977, -3.0634241, 3.1222591
1: -9.9893417, -6.7893000, -9.9652586, -6.7941055, -2.7990222, 2.7851758
2: -4.8524537, -1.6993924, -4.8349714, -1.6961858, -2.8448935, 2.8219218
3: -1.7331471, 1.6657543, -1.7141991, 1.6565455, -3.3896928, 3.3794432
4: -14.0059671, -10.0273829, -13.9962397, -10.0422916, -3.2620134, 3.2779865
5: -8.5547829, -5.0939217, -8.5526123, -5.1082077, -2.1326013, 2.1622021
6: -12.7819691, -8.5392828, -12.7700939, -8.5455151, -3.3041968, 3.2810135
7: -9.2108507, -5.6968465, -9.1866436, -5.7029181, -2.8444891, 2.8432083
8: 9.6396198, 12.5905581, 9.6531258, 12.5795822, -2.6725221, 2.6662409
9: -7.9759417, -3.6882730, -7.9701238, -3.7012701, -2.9967194, 3.0094361

Time for backsubstitution: 14.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6253
type: A, layer: 1, pos: 5762
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 5746
type: A, layer: 1, pos: 4630
type: A, layer: 1, pos: 832

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6253

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4918572, upper bound: 1.4878479
time: 31.23 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.4924293, upper bound: 1.4902745
time: 10.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 56.41 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4848586, upper bound: 1.4900393
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4848586, upper bound: 1.4900387
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4872792, upper bound: 1.4906092
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4872792, upper bound: 1.4906091
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4881537, upper bound: 1.4912670
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4881537, upper bound: 1.4912663
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4905768, upper bound: 1.4918403
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4905768, upper bound: 1.4918391
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4885617, upper bound: 1.4866129
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4891294, upper bound: 1.4890414
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4885619, upper bound: 1.4910286
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4891294, upper bound: 1.4934560
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4918572, upper bound: 1.4878479
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 56.41
Output dim: 8, lower bound: -1.4924293, upper bound: 1.4902745
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4924316, upper bound: 1.4946892
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4895523, upper bound: 1.4900386
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4919729, upper bound: 1.4906099
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4928463, upper bound: 1.4912663
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4952706, upper bound: 1.4918399
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4913911, upper bound: 1.4890432
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4913911, upper bound: 1.4934575
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4902788
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 56.41
Output dim: 8, lower bound: -1.4946891, upper bound: 1.4946914
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.6844100952148438
rel_dist={8: [-1.4952889722192388, 1.4952881994988214]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2446.41 seconds
