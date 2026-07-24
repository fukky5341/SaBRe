## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.6259627037


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305)
1: (-4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668)
2: (-6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820)
3: (-5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251)
4: (-6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038)
5: (-4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198)
6: (-4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974)
7: (-5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517)
8: (-6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594)
9: (-4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.07 + 5.11 = 6.18 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6335958, upper bound: 7.6335962

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6333416, upper bound: 7.6329650
time: 3.38 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329440, upper bound: 7.6329440
time: 2.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 5.92 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 5.92
Output dim: 2, lower bound: -7.6333416, upper bound: 7.6329650
NS_A2, status: Status.UNKNOWN, split count: 1, time: 5.92
Output dim: 2, lower bound: -7.6329440, upper bound: 7.6329440

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -4.2686481, 3.3034801, -4.8439870, 3.7249637, -7.9936118, 8.1474667
1: -3.3155420, 3.0498059, -3.7795856, 3.4268079, -6.7423496, 6.8293915
2: -5.4949217, 2.3689001, -6.2778797, 2.6069093, -8.1018314, 8.6467800
3: -4.8110600, 2.4704089, -5.4946775, 2.7603803, -7.5714402, 7.9650865
4: -5.1011324, 3.2392945, -5.7783298, 3.6380610, -8.7391930, 9.0176239
5: -3.8673596, 3.4099646, -4.3622389, 3.8365352, -7.7038946, 7.7722034
6: -4.0397010, 3.4505873, -4.5767317, 3.9007261, -7.9404268, 8.0273190
7: -4.6541395, 3.4808147, -5.2714081, 3.9153130, -8.5694523, 8.7522230
8: -5.3186646, 3.2162716, -6.0306888, 3.6089582, -8.9276228, 9.2469606
9: -3.7986977, 4.3246951, -4.2714772, 4.8971300, -8.6958275, 8.5961723

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317630, upper bound: 7.6304837
time: 10.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6329432, upper bound: 7.6325742
time: 3.56 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.3132782, 6.2759089, -4.7834363, 3.6805861, -11.9938641, 11.0593452
1: -6.5844011, 5.6741643, -3.7305417, 3.3870606, -9.9714622, 9.4047060
2: -10.9843225, 4.1436543, -6.1959019, 2.5812473, -13.5655699, 10.3395557
3: -9.5884886, 4.5238266, -5.4231091, 2.7295809, -12.3180695, 9.9469357
4: -9.8131104, 6.0509901, -5.7074418, 3.5959396, -13.4090500, 11.7584324
5: -7.3376122, 6.4407911, -4.3102756, 3.7915919, -11.1292038, 10.7510662
6: -7.8093801, 6.6083603, -4.5202670, 3.8534420, -11.6628218, 11.1286278
7: -8.9933872, 6.5240011, -5.2064905, 3.8693681, -12.8627548, 11.7304916
8: -10.3325996, 5.9963574, -5.9557323, 3.5676150, -13.9002151, 11.9520893
9: -7.1123838, 8.2964840, -4.2217007, 4.8374529, -11.9498367, 12.5181847

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314270, upper bound: 7.6304627
time: 4.16 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325541, upper bound: 7.6325542
time: 2.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.83 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 2, lower bound: -7.6317630, upper bound: 7.6304837
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 2, lower bound: -7.6329432, upper bound: 7.6325742
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 2, lower bound: -7.6314270, upper bound: 7.6304627
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 2, lower bound: -7.6325541, upper bound: 7.6325542

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2.0544791, 1.6642331, -0.6327204, 0.6538154, -2.7082944, 2.2969534
1: -1.6587214, 1.5523050, -0.6122586, 0.6269299, -2.2856512, 2.1645637
2: -2.3107984, 1.4857168, -0.1243145, 1.1261820, -3.4369802, 1.6100314
3: -2.1744158, 1.3505430, -0.4816597, 0.6408203, -2.8152361, 1.8322027
4: -2.4259250, 1.6757801, -0.7189755, 0.6833813, -3.1093063, 2.3947556
5: -1.8946854, 1.7060803, -0.6090171, 0.7215949, -2.6162803, 2.3150973
6: -1.9384396, 1.7116561, -0.5798560, 0.6723020, -2.6107416, 2.2915120
7: -2.2237339, 1.7968422, -0.6512892, 0.7386155, -2.9623494, 2.4481316
8: -2.5171628, 1.7071279, -0.7206078, 0.7722126, -3.2893753, 2.4277358
9: -1.9137287, 2.1263843, -0.6718641, 0.7479298, -2.6616585, 2.7982483

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304686, upper bound: 7.5921574
time: 4.56 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306432, upper bound: 7.5921658
time: 4.86 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.9733973, 3.0839028, -3.6603279, 2.8497655, -6.8231630, 6.7442307
1: -3.0780487, 2.8561590, -2.8303285, 2.6510715, -5.7291203, 5.6864872
2: -5.0836267, 2.2324247, -4.6422482, 2.0604300, -7.1440568, 6.8746729
3: -4.4626904, 2.3210492, -4.0982494, 2.1634922, -6.6261826, 6.4192986
4: -4.7573214, 3.0308218, -4.4014063, 2.8090172, -7.5663385, 7.4322281
5: -3.6105978, 3.1864536, -3.3333907, 2.9390576, -6.5496554, 6.5198441
6: -3.7670946, 3.2184243, -3.4833982, 2.9751120, -6.7422066, 6.7018223
7: -4.3368220, 3.2556291, -3.9991078, 3.0176325, -7.3544545, 7.2547369
8: -4.9507952, 3.0113065, -4.5607572, 2.8001130, -7.7509079, 7.5720634
9: -3.5559435, 4.0345802, -3.3003826, 3.7390604, -7.2950039, 7.3349628

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6322382, upper bound: 7.6319984
time: 3.31 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6325112, upper bound: 7.6320795
time: 4.35 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.9475546, 4.5396948, -0.6061128, 0.6363257, -6.5838804, 5.1458077
1: -4.6825829, 4.1357999, -0.5976955, 0.6135504, -5.2961330, 4.7334952
2: -7.7880030, 3.1085556, -0.0878957, 1.1216575, -8.9096603, 3.1964512
3: -6.7936106, 3.3253481, -0.4566112, 0.6298405, -7.4234509, 3.7819593
4: -7.0588965, 4.4039311, -0.6896520, 0.6657025, -7.7245989, 5.0935831
5: -5.3121061, 4.6658125, -0.5910856, 0.7027930, -6.0148993, 5.2568979
6: -5.6150656, 4.7624488, -0.5592645, 0.6559900, -6.2710557, 5.3217134
7: -6.4609408, 4.7470765, -0.6287217, 0.7235727, -7.1845136, 5.3757982
8: -7.3980551, 4.3743696, -0.6983821, 0.7564668, -8.1545219, 5.0727520
9: -5.1743474, 5.9693584, -0.6531427, 0.7232587, -5.8976059, 6.6225014

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6300484, upper bound: 7.5921343
time: 7.41 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302708, upper bound: 7.5921424
time: 2.34 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.0024185, 6.0469885, -3.5986392, 2.8047452, -10.8071632, 9.6456280
1: -6.3337159, 5.4724278, -2.7810116, 2.6103392, -8.9440556, 8.2534389
2: -10.5640163, 4.0049324, -4.5582352, 2.0346484, -12.5986652, 8.5631676
3: -9.2221451, 4.3655596, -4.0249224, 2.1329720, -11.3551168, 8.3904819
4: -9.4531078, 5.8331246, -4.3285012, 2.7667048, -12.2198124, 10.1616259
5: -7.0714955, 6.2077842, -3.2799673, 2.8933988, -9.9648943, 9.4877510
6: -7.5212197, 6.3655143, -3.4265873, 2.9268625, -10.4480820, 9.7921019
7: -8.6606464, 6.2903137, -3.9330668, 2.9709082, -11.6315546, 10.2233810
8: -9.9460192, 5.7810035, -4.4842110, 2.7588148, -12.7048340, 10.2652149
9: -6.8580933, 7.9917693, -3.2494252, 3.6777515, -10.5358448, 11.2411947

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317888, upper bound: 7.6319766
time: 2.18 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320576, upper bound: 7.6320576
time: 3.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6304686, upper bound: 7.5921574
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6306432, upper bound: 7.5921658
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6322382, upper bound: 7.6319984
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6325112, upper bound: 7.6320795
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6300484, upper bound: 7.5921343
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6302708, upper bound: 7.5921424
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6317888, upper bound: 7.6319766
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.04
Output dim: 2, lower bound: -7.6320576, upper bound: 7.6320576

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.4375579, 0.5057535, -0.3759645, 0.4386618, -0.8762197, 0.8817180
1: -0.4761628, 0.5051691, -0.4183383, 0.4453653, -0.9215281, 0.9235073
2: 0.1516788, 1.0878110, 0.2479169, 1.0706682, -0.9189894, 0.8398941
3: -0.3131579, 0.5396056, -0.2651235, 0.4927065, -0.8058645, 0.8047291
4: -0.5113816, 0.5311555, -0.4527672, 0.4586921, -0.9700737, 0.9839227
5: -0.4673158, 0.5427294, -0.4089996, 0.4708560, -0.9381718, 0.9517289
6: -0.4112628, 0.5450190, -0.3561119, 0.4853476, -0.8966104, 0.9011309
7: -0.4692430, 0.5872211, -0.4098575, 0.5138015, -0.9830444, 0.9970787
8: -0.5345352, 0.6277806, -0.4724873, 0.5635416, -1.0980768, 1.1002679
9: -0.5218457, 0.5510181, -0.4643419, 0.4690111, -0.9908568, 1.0153600

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286447, upper bound: 7.4237966
time: 2.28 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287959, upper bound: 7.4420110
time: 5.00 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9222797, 0.8429014, -0.3931107, 0.4580549, -1.3803346, 1.2360121
1: -0.8082142, 0.7876428, -0.4340487, 0.4617651, -1.2699792, 1.2216915
2: -0.5528372, 1.1784894, 0.2201818, 1.0747434, -1.6275806, 0.9583076
3: -0.8042040, 0.7677456, -0.2785083, 0.5059881, -1.3101921, 1.0462539
4: -1.0633701, 0.8693500, -0.4686782, 0.4791518, -1.5425220, 1.3380282
5: -0.8503823, 0.9127451, -0.4257610, 0.4905806, -1.3409629, 1.3385061
6: -0.8323878, 0.8622503, -0.3718149, 0.5017007, -1.3340886, 1.2340653
7: -0.9543545, 0.9268708, -0.4262078, 0.5346959, -1.4890504, 1.3530786
8: -1.0166404, 0.9484719, -0.4888583, 0.5811551, -1.5977955, 1.4373301
9: -0.9048309, 1.0035747, -0.4799495, 0.4924874, -1.3973184, 1.4835242

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293055, upper bound: 7.4238101
time: 3.58 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6294149, upper bound: 7.4420255
time: 3.16 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6240726, 1.3434670, -2.5611894, 2.0513279, -3.6754005, 3.9046564
1: -1.3315597, 1.2480187, -2.0364418, 1.9174681, -3.2490277, 3.2844605
2: -1.6417361, 1.3585764, -3.0912013, 1.6502118, -3.2919478, 4.4497776
3: -1.6529676, 1.1216012, -2.7892103, 1.6188232, -3.2717907, 3.9108114
4: -1.9091322, 1.3627229, -3.0788908, 2.0503390, -3.9594712, 4.4416137
5: -1.4943283, 1.3903141, -2.3653483, 2.0939534, -3.5882816, 3.7556624
6: -1.4967098, 1.3748986, -2.4584308, 2.1190734, -3.6157832, 3.8333292
7: -1.7411511, 1.4582176, -2.8043394, 2.1896694, -3.9308205, 4.2625570
8: -1.9237050, 1.4178429, -3.2000084, 2.0574296, -3.9811344, 4.6178513
9: -1.5208721, 1.6871395, -2.3791962, 2.6529346, -4.1738067, 4.0663357

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315136, upper bound: 7.6311990
time: 2.45 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315409, upper bound: 7.6313845
time: 3.37 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.6027346, 2.0802090, -2.8696129, 2.2713099, -4.8740444, 4.9498219
1: -2.0643280, 1.9450513, -2.2473054, 2.1226027, -4.1869307, 4.1923566
2: -3.1670213, 1.6637118, -3.5330350, 1.7515802, -4.9186015, 5.1967468
3: -2.8332653, 1.6402210, -3.1437230, 1.7713864, -4.6046515, 4.7839441
4: -3.1266012, 2.0799649, -3.4531260, 2.2621088, -5.3887100, 5.5330906
5: -2.4012480, 2.1384859, -2.6349540, 2.3312769, -4.7325249, 4.7734399
6: -2.4963412, 2.1501756, -2.7461295, 2.3565311, -4.8528724, 4.8963051
7: -2.8478332, 2.2213492, -3.1390204, 2.4175413, -5.2653742, 5.3603697
8: -3.2487485, 2.0868356, -3.5804396, 2.2638538, -5.5126023, 5.6672754
9: -2.4127998, 2.6921189, -2.6388662, 2.9542093, -5.3670092, 5.3309851

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319107, upper bound: 7.6312999
time: 3.45 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319687, upper bound: 7.6315047
time: 2.54 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.5589991, 2.7780263, -0.3679520, 0.4299458, -3.9889450, 3.1459785
1: -2.7609611, 2.5787771, -0.4111218, 0.4376045, -3.1985655, 2.9898989
2: -4.5058522, 2.0404634, 0.2604243, 1.0684789, -5.5743313, 1.7800391
3: -3.9786603, 2.1189060, -0.2588759, 0.4866097, -4.4652700, 2.3777819
4: -4.2717552, 2.7448380, -0.4452316, 0.4492521, -4.7210073, 3.1900697
5: -3.2518902, 2.8606019, -0.4016528, 0.4615302, -3.7134204, 3.2622547
6: -3.4003422, 2.8951626, -0.3490357, 0.4775978, -3.8779399, 3.2441983
7: -3.8963063, 2.9406056, -0.4022073, 0.5042163, -4.4005227, 3.3428130
8: -4.4369383, 2.7339518, -0.4644707, 0.5555166, -4.9924550, 3.1984224
9: -3.2147672, 3.6201253, -0.4568843, 0.4585291, -3.6732965, 4.0770097

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6089962, upper bound: 7.4237736
time: 2.71 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6200372, upper bound: 7.4419872
time: 2.50 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.4387751, 3.4268723, -0.3849972, 0.4492262, -4.8880014, 3.8118694
1: -3.4659214, 3.1547296, -0.4267410, 0.4539097, -3.9198310, 3.5814705
2: -5.7159557, 2.4193392, 0.2328508, 1.0725321, -6.7884879, 2.1864884
3: -5.0219154, 2.5579576, -0.2721821, 0.4998149, -5.5217304, 2.8301399
4: -5.3099284, 3.3515816, -0.4610519, 0.4695930, -5.7795215, 3.8126335
5: -4.0128536, 3.5236270, -0.4183160, 0.4811420, -4.4939957, 3.9419432
6: -4.2143545, 3.5845549, -0.3646470, 0.4938565, -4.7082109, 3.9492018
7: -4.8394384, 3.6060505, -0.4184621, 0.5249913, -5.3644300, 4.0245128
8: -5.5272703, 3.3279989, -0.4807488, 0.5730268, -6.1002970, 3.8087478
9: -3.9385438, 4.4950681, -0.4724032, 0.4818683, -4.4204121, 4.9674711

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285847, upper bound: 7.4237860
time: 4.86 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287830, upper bound: 7.4420011
time: 5.45 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.4747863, 4.1956425, -2.5134125, 2.0175185, -7.4923048, 6.7090549
1: -4.2982330, 3.8287520, -2.0037293, 1.8853480, -6.1835809, 5.8324814
2: -7.1613092, 2.8987954, -3.0233560, 1.6344966, -8.7958059, 5.9221516
3: -6.2395878, 3.0862188, -2.7364335, 1.5950335, -7.8346214, 5.8226523
4: -6.5063496, 4.0839391, -3.0209806, 2.0171900, -8.5235395, 7.1049194
5: -4.9062939, 4.3081031, -2.3234725, 2.0581331, -6.9644270, 6.6315756
6: -5.1734214, 4.3960547, -2.4139771, 2.0824580, -7.2558794, 6.8100319
7: -5.9546199, 4.3898392, -2.7523909, 2.1545715, -8.1091919, 7.1422300
8: -6.8181944, 4.0572567, -3.1409383, 2.0256248, -8.8438187, 7.1981950
9: -4.7843189, 5.5114660, -2.3387520, 2.6069543, -7.3912735, 7.8502178

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309685, upper bound: 7.6311769
time: 2.34 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309910, upper bound: 7.6313600
time: 2.21 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.5732942, 4.9984035, -2.8079538, 2.2269444, -8.8002386, 7.8063574
1: -5.1796169, 4.5439219, -2.2048202, 2.0812802, -7.2608972, 6.7487421
2: -8.6435480, 3.3709190, -3.4461098, 1.7310233, -10.3745708, 6.8170290
3: -7.5380139, 3.6394167, -3.0683122, 1.7407813, -9.2787952, 6.7077289
4: -7.7917151, 4.8401198, -3.3783178, 2.2194538, -10.0111694, 8.2184372
5: -5.8471150, 5.1372399, -2.5808668, 2.2844152, -8.1315308, 7.7181067
6: -6.1933317, 5.2504425, -2.6886446, 2.3084738, -8.5018053, 7.9390869
7: -7.1297507, 5.2142563, -3.0719848, 2.3709774, -9.5007286, 8.2862415
8: -8.1741972, 4.7977457, -3.5041375, 2.2218099, -10.3960075, 8.3018837
9: -5.6849499, 6.5934253, -2.5868192, 2.8926830, -8.5776329, 9.1802444

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314259, upper bound: 7.6312818
time: 6.05 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314853, upper bound: 7.6314852
time: 2.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 9.52 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6286447, upper bound: 7.4237966
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6287959, upper bound: 7.4420110
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6293055, upper bound: 7.4238101
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6294149, upper bound: 7.4420255
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6315136, upper bound: 7.6311990
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6315409, upper bound: 7.6313845
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6319107, upper bound: 7.6312999
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6319687, upper bound: 7.6315047
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6089962, upper bound: 7.4237736
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6200372, upper bound: 7.4419872
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6285847, upper bound: 7.4237860
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6287830, upper bound: 7.4420011
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6309685, upper bound: 7.6311769
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6309910, upper bound: 7.6313600
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6314259, upper bound: 7.6312818
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 9.52
Output dim: 2, lower bound: -7.6314853, upper bound: 7.6314852

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.3215374, 0.3785565, -0.1655774, 0.1786179, -0.5001553, 0.5441339
1: -0.3690412, 0.3930303, -0.2128506, 0.2244805, -0.5935217, 0.6058809
2: 0.3340648, 1.0567021, 0.5842205, 1.0453693, -0.7113044, 0.4724817
3: -0.2226866, 0.4510706, -0.0841027, 0.2959728, -0.5186595, 0.5351733
4: -0.4020062, 0.3943269, -0.2298062, 0.2244284, -0.6264346, 0.6241331
5: -0.3577445, 0.4080103, -0.2075235, 0.2276342, -0.5853786, 0.6155338
6: -0.3073493, 0.4330929, -0.1673270, 0.2388721, -0.5462214, 0.6004199
7: -0.3579624, 0.4483341, -0.1962765, 0.2546783, -0.6126407, 0.6446106
8: -0.4192457, 0.5084815, -0.2243754, 0.3214177, -0.7406634, 0.7328569
9: -0.4142912, 0.3965317, -0.2364042, 0.2089900, -0.6232812, 0.6329359

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3278260, upper bound: 7.0120853
time: 4.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2280558, upper bound: 7.0119845
time: 3.38 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.3260155, 0.3836042, -0.1901525, 0.2063102, -0.5323257, 0.5737567
1: -0.3731359, 0.3973123, -0.2399131, 0.2529218, -0.6260577, 0.6372254
2: 0.3268425, 1.0577728, 0.5435508, 1.0462335, -0.7193910, 0.5142220
3: -0.2261810, 0.4545308, -0.1065318, 0.3254724, -0.5516534, 0.5610626
4: -0.4061590, 0.3996609, -0.2587124, 0.2485973, -0.6547563, 0.6583732
5: -0.3621021, 0.4131572, -0.2328232, 0.2589746, -0.6210767, 0.6459804
6: -0.3114386, 0.4373646, -0.1857919, 0.2771498, -0.5885884, 0.6231564
7: -0.3622308, 0.4537770, -0.2199289, 0.2859460, -0.6481768, 0.6737059
8: -0.4235272, 0.5130725, -0.2591266, 0.3541500, -0.7776772, 0.7721992
9: -0.4183669, 0.4026374, -0.2667136, 0.2394815, -0.6578484, 0.6693509

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3380506, upper bound: 7.0288897
time: 2.94 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2380407, upper bound: 7.0287932
time: 6.21 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.4921039, 0.5650793, -0.1671329, 0.1802153, -0.6723191, 0.7322122
1: -0.5289126, 0.5527412, -0.2145346, 0.2263891, -0.7553017, 0.7672758
2: 0.0643053, 1.0998832, 0.5815797, 1.0453516, -0.9810463, 0.5183035
3: -0.3522720, 0.5818375, -0.0853313, 0.2980142, -0.6502862, 0.6671689
4: -0.5708414, 0.5884474, -0.2316887, 0.2257293, -0.7965707, 0.8201361
5: -0.5220425, 0.6096491, -0.2092742, 0.2295879, -0.7516304, 0.8189233
6: -0.4672987, 0.5908822, -0.1685476, 0.2411903, -0.7084889, 0.7594298
7: -0.5257089, 0.6498010, -0.1975447, 0.2567992, -0.7825081, 0.8473457
8: -0.5927641, 0.6878866, -0.2263505, 0.3236338, -0.9163979, 0.9142370
9: -0.5707085, 0.6236483, -0.2383775, 0.2107961, -0.7815046, 0.8620258

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3758649, upper bound: 7.0121016
time: 3.20 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2622634, upper bound: 7.0119943
time: 4.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.5505497, 0.6035465, -0.2087835, 0.2307984, -0.7813481, 0.8123299
1: -0.5662493, 0.5851125, -0.2580642, 0.2748550, -0.8411042, 0.8431766
2: -0.0169831, 1.1109116, 0.5157579, 1.0462120, -1.0631950, 0.5951537
3: -0.4047529, 0.6095151, -0.1246156, 0.3462098, -0.7509627, 0.7341306
4: -0.6348512, 0.6277246, -0.2792440, 0.2672769, -0.9021281, 0.9069686
5: -0.5581580, 0.6628464, -0.2513576, 0.2799813, -0.8381393, 0.9142039
6: -0.5161189, 0.6252654, -0.2023629, 0.3020389, -0.8181578, 0.8276283
7: -0.5826356, 0.6913940, -0.2400438, 0.3074629, -0.8900986, 0.9314378
8: -0.6523023, 0.7245587, -0.2853622, 0.3759942, -1.0282965, 1.0099208
9: -0.6140503, 0.6774718, -0.2891697, 0.2616687, -0.8757190, 0.9666415

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3872771, upper bound: 7.0289056
time: 2.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2758285, upper bound: 7.0288028
time: 4.47 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.0956531, 0.9566654, -1.2009926, 1.0301166, -2.1257696, 2.1576581
1: -0.9337584, 0.8934159, -1.0100741, 0.9581370, -1.8918953, 1.9034901
2: -0.8110871, 1.2148153, -0.9754078, 1.2378676, -2.0489547, 2.1902232
3: -1.0121615, 0.8501461, -1.1392158, 0.9014394, -1.9136009, 1.9893619
4: -1.2694221, 0.9897659, -1.3962607, 1.0601850, -2.3296070, 2.3860266
5: -1.0051379, 1.0237787, -1.1012683, 1.0928749, -2.0980129, 2.1250470
6: -0.9914132, 0.9800230, -1.0916661, 1.0550939, -2.0465071, 2.0716891
7: -1.1477908, 1.0535733, -1.2657177, 1.1291806, -2.2769713, 2.3192911
8: -1.2098949, 1.0649045, -1.3418822, 1.1332012, -2.3430963, 2.4067867
9: -1.0522729, 1.1554426, -1.1393110, 1.2582780, -2.3105509, 2.2947536

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6283250
time: 3.10 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6311681
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.1488414, 0.9928353, -1.5339347, 1.2759864, -2.4248278, 2.5267701
1: -0.9724360, 0.9268545, -1.2617304, 1.1808023, -2.1532383, 2.1885848
2: -0.8927552, 1.2267164, -1.5018319, 1.3291522, -2.2219074, 2.7285483
3: -1.0762475, 0.8760806, -1.5441580, 1.0736284, -2.1498759, 2.4202385
4: -1.3337193, 1.0261621, -1.7992783, 1.2947860, -2.6285052, 2.8254404
5: -1.0531402, 1.0587689, -1.4100418, 1.3249989, -2.3781390, 2.4688106
6: -1.0410404, 1.0177692, -1.4129441, 1.3044155, -2.3454559, 2.4307132
7: -1.2073554, 1.0927880, -1.6390570, 1.3843610, -2.5917163, 2.7318449
8: -1.2735596, 1.0998133, -1.7980084, 1.3548675, -2.6284270, 2.8978219
9: -1.0970556, 1.2054052, -1.4364104, 1.5963377, -2.6933932, 2.6418157

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
time: 4.27 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5517384, upper bound: 7.4419719
time: 3.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.9732730, 1.6038033, -1.4180212, 1.1890336, -3.1623068, 3.0218244
1: -1.5965267, 1.4923103, -1.1727695, 1.1016750, -2.6982017, 2.6650798
2: -2.1917653, 1.4594777, -1.3183019, 1.2962567, -3.4880219, 2.7777796
3: -2.0742850, 1.3075709, -1.4024230, 1.0132606, -3.0875456, 2.7099938
4: -2.3261719, 1.6167358, -1.6583762, 1.2114739, -3.5376458, 3.2751122
5: -1.8212241, 1.6484563, -1.3012912, 1.2423633, -3.0635874, 2.9497476
6: -1.8570006, 1.6461083, -1.3009728, 1.2157518, -3.0727525, 2.9470811
7: -2.1325765, 1.7308207, -1.5070120, 1.2953182, -3.4278946, 3.2378325
8: -2.4028466, 1.6540153, -1.6387653, 1.2756732, -3.6785197, 3.2927806
9: -1.8369036, 2.0462775, -1.3308322, 1.4771473, -3.3140509, 3.3771098

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6282341
time: 2.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6313000
time: 4.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.1051967, 1.7020730, -1.8130972, 1.4855680, -3.5907648, 3.5151701
1: -1.6964428, 1.5865093, -1.4742785, 1.3773447, -3.0737877, 3.0607877
2: -2.3994079, 1.4982717, -1.9406636, 1.4102278, -3.8096356, 3.4389353
3: -2.2362418, 1.3777044, -1.8830147, 1.2208179, -3.4570599, 3.2607191
4: -2.4838617, 1.7124313, -2.1357846, 1.4981339, -3.9819956, 3.8482161
5: -1.9441192, 1.7480804, -1.6717939, 1.5273182, -3.4714375, 3.4198742
6: -1.9940128, 1.7491753, -1.6923141, 1.5212839, -3.5152967, 3.4414895
7: -2.2790396, 1.8346171, -1.9555091, 1.6024905, -3.8815303, 3.7901263
8: -2.5833616, 1.7426432, -2.1839557, 1.5447130, -4.1280746, 3.9265990
9: -1.9564302, 2.1810212, -1.6915098, 1.8835993, -3.8400295, 3.8725309

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
time: 4.70 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
time: 2.69 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.6776330, 2.8633990, -0.1634205, 0.1764030, -3.8540361, 3.0268195
1: -2.8502903, 2.6549621, -0.2105156, 0.2218339, -3.0721242, 2.8654776
2: -4.6704416, 2.0901721, 0.5878826, 1.0447383, -5.7151799, 1.5022895
3: -4.1177406, 2.1732159, -0.0823990, 0.2931424, -4.4108829, 2.2556150
4: -4.4131274, 2.8189101, -0.2271958, 0.2226245, -4.6357517, 3.0461059
5: -3.3544755, 2.9570847, -0.2050961, 0.2249250, -3.5794005, 3.1621807
6: -3.5080936, 2.9867780, -0.1656345, 0.2356572, -3.7437508, 3.1524124
7: -4.0226574, 3.0261099, -0.1945181, 0.2517378, -4.2743950, 3.2206280
8: -4.5789042, 2.8084664, -0.2216369, 0.3183449, -4.8972492, 3.0301034
9: -3.3112485, 3.7392657, -0.2336679, 0.2064860, -3.5177345, 3.9729335

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3453898, upper bound: 7.0120707
time: 3.52 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2414656, upper bound: 7.0119753
time: 4.26 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.8581655, 2.9957039, -0.2035037, 0.2232151, -4.0813808, 3.1992078
1: -2.9952476, 2.7734189, -0.2528629, 0.2685096, -3.2637572, 3.0262818
2: -4.9156423, 2.1640007, 0.5239875, 1.0458157, -5.9614582, 1.6400132
3: -4.3332272, 2.2631242, -0.1196031, 0.3402539, -4.6734810, 2.3827274
4: -4.6276579, 2.9428308, -0.2736076, 0.2612221, -4.8888798, 3.2164383
5: -3.5100358, 3.0905473, -0.2457275, 0.2742047, -3.7842405, 3.3362749
6: -3.6757510, 3.1281235, -0.1973287, 0.2949744, -3.9707253, 3.3254523
7: -4.2159624, 3.1625907, -0.2339984, 0.3014715, -4.5174341, 3.3965890
8: -4.8029947, 2.9293351, -0.2778364, 0.3696542, -5.1726489, 3.2071714
9: -3.4604352, 3.9192615, -0.2825514, 0.2553706, -3.7158058, 4.2018127

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3575303, upper bound: 7.0288756
time: 5.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2567094, upper bound: 7.0287855
time: 3.61 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -4.7718534, 3.6773858, -1.1628827, 1.0031489, -5.7750025, 4.8402686
1: -3.7295480, 3.3703914, -0.9822426, 0.9331744, -4.6627226, 4.3526340
2: -6.2022815, 2.5954933, -0.9160359, 1.2279017, -7.4301834, 3.5115292
3: -5.4078465, 2.7274580, -1.0935287, 0.8823293, -6.2901759, 3.8209867
4: -5.6844730, 3.5928669, -1.3500848, 1.0339899, -6.7184629, 4.9429517
5: -4.3040648, 3.7830396, -1.0665252, 1.0674635, -5.3715281, 4.8495646
6: -4.5192666, 3.8471582, -1.0561177, 1.0272465, -5.5465131, 4.9032760
7: -5.2007318, 3.8579092, -1.2233033, 1.1003927, -6.3011246, 5.0812125
8: -5.9438753, 3.5751503, -1.2902727, 1.1081519, -7.0520272, 4.8654232
9: -4.2081399, 4.8166890, -1.1070288, 1.2204630, -5.4286032, 5.9237180

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
time: 4.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311509
time: 4.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -4.8307295, 3.7204826, -1.4929535, 1.2450616, -6.0757914, 5.2134361
1: -3.7762842, 3.4090035, -1.2301675, 1.1522005, -4.9284849, 4.6391711
2: -6.2812252, 2.6188145, -1.4371502, 1.3172511, -7.5984764, 4.0559645
3: -5.4783545, 2.7568951, -1.4943238, 1.0521894, -6.5305438, 4.2512188
4: -5.7546797, 3.6332157, -1.7495033, 1.2649043, -7.0195837, 5.3827190
5: -4.3544202, 3.8268292, -1.3713409, 1.2955949, -5.6500149, 5.1981702
6: -4.5738249, 3.8931198, -1.3735831, 1.2727892, -5.8466139, 5.2667027
7: -5.2637720, 3.9021881, -1.5921925, 1.3525438, -6.6163158, 5.4943805
8: -6.0169573, 3.6140194, -1.7415417, 1.3267570, -7.3437142, 5.3555613
9: -4.2567363, 4.8756843, -1.3987327, 1.5542634, -5.8109999, 6.2744169

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
time: 2.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313076
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.7915745, 4.4235950, -1.3771781, 1.1589034, -6.9504776, 5.8007731
1: -4.5464697, 4.0345554, -1.1416976, 1.0740740, -5.6205435, 5.1762533
2: -7.5860028, 3.0361753, -1.2538208, 1.2846625, -8.8706656, 4.2899961
3: -6.6116295, 3.2410007, -1.3523948, 0.9921165, -7.6037459, 4.5933952
4: -6.8767323, 4.2952223, -1.6088462, 1.1824472, -8.0591793, 5.9040685
5: -5.1787548, 4.5547438, -1.2636000, 1.2135365, -6.3922911, 5.8183436
6: -5.4647703, 4.6403966, -1.2616413, 1.1848114, -6.6495819, 5.9020376
7: -6.2922392, 4.6238313, -1.4614695, 1.2636461, -7.5558853, 6.0853009
8: -7.2029243, 4.2633080, -1.5828717, 1.2480716, -8.4509954, 5.8461800
9: -5.0431585, 5.8210378, -1.2941175, 1.4359045, -6.4790630, 7.1151552

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
time: 6.96 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312817
time: 3.40 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -5.9926887, 4.5704522, -1.7683985, 1.4520612, -7.4447498, 6.3388510
1: -4.7087898, 4.1658034, -1.4402798, 1.3452897, -6.0540795, 5.6060834
2: -7.8552809, 3.1182122, -1.8708315, 1.3969896, -9.2522707, 4.9890437
3: -6.8510852, 3.3427217, -1.8287210, 1.1967047, -8.0477896, 5.1714430
4: -7.1139984, 4.4339471, -2.0820878, 1.4649547, -8.5789528, 6.5160351
5: -5.3504949, 4.7038383, -1.6293851, 1.4950441, -6.8455391, 6.3332233
6: -5.6523695, 4.7972584, -1.6457657, 1.4859622, -7.1383314, 6.4430242
7: -6.5074272, 4.7751551, -1.9053166, 1.5667646, -8.0741920, 6.6804714
8: -7.4520969, 4.3985987, -2.1217227, 1.5141705, -8.9662676, 6.5203214
9: -5.2086983, 6.0206003, -1.6505555, 1.8376806, -7.0463791, 7.6711559

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287827
time: 2.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314848
time: 2.83 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.17 seconds
NS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.3278260, upper bound: 7.0120853
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.2280558, upper bound: 7.0119845
NS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.3380506, upper bound: 7.0288897
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.2380407, upper bound: 7.0287932
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.3758649, upper bound: 7.0121016
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.2622634, upper bound: 7.0119943
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.3872771, upper bound: 7.0289056
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.2758285, upper bound: 7.0288028
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6283250
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5459815, upper bound: 7.6311681
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5517384, upper bound: 7.4419719
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6282341
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5391196, upper bound: 7.6313000
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.3453898, upper bound: 7.0120707
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.2414656, upper bound: 7.0119753
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.3575303, upper bound: 7.0288756
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.2567094, upper bound: 7.0287855
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311509
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313076
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312817
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287827
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.17
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314848

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0999603, 0.1069386, -1.2009926, 1.0301166, -1.1300769, 1.3079312
1: -0.1372842, 0.1415909, -1.0100741, 0.9581370, -1.0954212, 1.1516651
2: 0.7027340, 1.0396814, -0.9754078, 1.2378676, -0.5351336, 2.0150893
3: -0.0289697, 0.2068110, -1.1392158, 0.9014394, -0.9304091, 1.3460268
4: -0.1513371, 0.1660503, -1.3962607, 1.0601850, -1.2115221, 1.5623109
5: -0.1289959, 0.1403148, -1.1012683, 1.0928749, -1.2218708, 1.2415831
6: -0.1125522, 0.1390101, -1.0916661, 1.0550939, -1.1676461, 1.2306762
7: -0.1393690, 0.1629245, -1.2657177, 1.1291806, -1.2685496, 1.4286423
8: -0.1359288, 0.2251918, -1.3418822, 1.1332012, -1.2691300, 1.5670741
9: -0.1500749, 0.1279543, -1.1393110, 1.2582780, -1.4083530, 1.2672652

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
time: 7.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
time: 7.44 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.5137417, 0.5817116, -1.2009926, 1.0301166, -1.5438583, 1.7827042
1: -0.5447940, 0.5669526, -1.0100741, 0.9581370, -1.5029309, 1.5770267
2: 0.0311265, 1.1043226, -0.9754078, 1.2378676, -1.2067411, 2.0797305
3: -0.3708899, 0.5945812, -1.1392158, 0.9014394, -1.2723293, 1.7337971
4: -0.5962261, 0.6047728, -1.3962607, 1.0601850, -1.6564111, 2.0010335
5: -0.5363327, 0.6336310, -1.1012683, 1.0928749, -1.6292076, 1.7348993
6: -0.4864335, 0.6049811, -1.0916661, 1.0550939, -1.5415274, 1.6966472
7: -0.5493956, 0.6686618, -1.2657177, 1.1291806, -1.6785762, 1.9343796
8: -0.6184307, 0.7035065, -1.3418822, 1.1332012, -1.7516320, 2.0453887
9: -0.5875289, 0.6464775, -1.1393110, 1.2582780, -1.8458070, 1.7857884

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6311681
time: 3.08 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5459817, upper bound: 7.4237516
time: 3.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0997177, 0.1066720, -1.5339347, 1.2759864, -1.3757041, 1.6406066
1: -0.1370031, 0.1412840, -1.2617304, 1.1808023, -1.3178054, 1.4030144
2: 0.7031748, 1.0396798, -1.5018319, 1.3291522, -0.6259775, 2.5415115
3: -0.0287646, 0.2064803, -1.5441580, 1.0736284, -1.1023930, 1.7506384
4: -0.1510483, 0.1658331, -1.7992783, 1.2947860, -1.4458343, 1.9651114
5: -0.1287037, 0.1399902, -1.4100418, 1.3249989, -1.4537026, 1.5500320
6: -0.1123485, 0.1386405, -1.4129441, 1.3044155, -1.4167640, 1.5515846
7: -0.1391573, 0.1625858, -1.6390570, 1.3843610, -1.5235183, 1.8016429
8: -0.1355999, 0.2248350, -1.7980084, 1.3548675, -1.4904673, 2.0228434
9: -0.1497552, 0.1276529, -1.4364104, 1.5963377, -1.7460929, 1.5640633

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
time: 2.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
time: 3.53 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.1481730, 0.1599290, -1.4180212, 1.1890336, -1.3372066, 1.5779502
1: -0.1931481, 0.2025903, -1.1727695, 1.1016750, -1.2948232, 1.3753598
2: 0.6151205, 1.0425050, -1.3183019, 1.2962567, -0.6811361, 2.3608069
3: -0.0697279, 0.2725243, -1.4024230, 1.0132606, -1.0829885, 1.6749474
4: -0.2087196, 0.2092075, -1.6583762, 1.2114739, -1.4201934, 1.8675838
5: -0.1870432, 0.2048456, -1.3012912, 1.2423633, -1.4294065, 1.5061369
6: -0.1530457, 0.2124394, -1.3009728, 1.2157518, -1.3687974, 1.5134122
7: -0.1814390, 0.2302229, -1.5070120, 1.2953182, -1.4767573, 1.7372348
8: -0.2012803, 0.2961308, -1.6387653, 1.2756732, -1.4769534, 1.9348962
9: -0.2136270, 0.1878617, -1.3308322, 1.4771473, -1.6907743, 1.5186939

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282343
time: 3.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282341
time: 3.12 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.3306952, 1.1253389, -1.4180212, 1.1890336, -2.5197287, 2.5433602
1: -1.1061755, 1.0433774, -1.1727695, 1.1016750, -2.2078505, 2.2161469
2: -1.1810069, 1.2723873, -1.3183019, 1.2962567, -2.4772635, 2.5906892
3: -1.2967119, 0.9677966, -1.4024230, 1.0132606, -2.3099725, 2.3702197
4: -1.5545385, 1.1491244, -1.6583762, 1.2114739, -2.7660124, 2.8075006
5: -1.2197224, 1.1831168, -1.3012912, 1.2423633, -2.4620857, 2.4844079
6: -1.2164470, 1.1506547, -1.3009728, 1.2157518, -2.4321988, 2.4516275
7: -1.4102179, 1.2283047, -1.5070120, 1.2953182, -2.7055361, 2.7353168
8: -1.5208342, 1.2165208, -1.6387653, 1.2756732, -2.7965074, 2.8552861
9: -1.2542810, 1.3892401, -1.3308322, 1.4771473, -2.7314284, 2.7200723

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5391198, upper bound: 7.4237475
time: 2.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6313000
time: 3.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1587485, 0.1715083, -1.8130972, 1.4855680, -1.6443166, 1.9846056
1: -0.2053554, 0.2160324, -1.4742785, 1.3773447, -1.5827001, 1.6903110
2: 0.5959755, 1.0424958, -1.9406636, 1.4102278, -0.8142523, 2.9831595
3: -0.0786342, 0.2869375, -1.8830147, 1.2208179, -1.2994521, 2.1699522
4: -0.2215261, 0.2186382, -2.1357846, 1.4981339, -1.7196600, 2.3544228
5: -0.1997317, 0.2189467, -1.6717939, 1.5273182, -1.7270499, 1.8907406
6: -0.1618941, 0.2286291, -1.6923141, 1.5212839, -1.6831779, 1.9209433
7: -0.1906322, 0.2452627, -1.9555091, 1.6024905, -1.7931228, 2.2007718
8: -0.2155851, 0.3116325, -2.1839557, 1.5447130, -1.7602981, 2.4955883
9: -0.2276510, 0.2009524, -1.6915098, 1.8835993, -2.1112502, 1.8924623

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
time: 3.34 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
time: 6.47 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.4588451, 1.2203939, -1.8130972, 1.4855680, -2.9444132, 3.0334911
1: -1.2039182, 1.1288085, -1.4742785, 1.3773447, -2.5812631, 2.6030869
2: -1.3840015, 1.3078859, -1.9406636, 1.4102278, -2.7942293, 3.2485495
3: -1.4534780, 1.0339497, -1.8830147, 1.2208179, -2.6742959, 2.9169645
4: -1.7099203, 1.2396287, -2.1357846, 1.4981339, -3.2080541, 3.3754134
5: -1.3381892, 1.2731776, -1.6717939, 1.5273182, -2.8655076, 2.9449716
6: -1.3403305, 1.2472912, -1.6923141, 1.5212839, -2.8616142, 2.9396052
7: -1.5539688, 1.3266928, -1.9555091, 1.6024905, -3.1564593, 3.2822018
8: -1.6963817, 1.3024652, -2.1839557, 1.5447130, -3.2410946, 3.4864209
9: -1.3689091, 1.5193797, -1.6915098, 1.8835993, -3.2525084, 3.2108896

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
time: 2.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.0393264, 0.9178565, -1.1628827, 1.0031489, -2.0424752, 2.0807393
1: -0.8928415, 0.8542848, -0.9822426, 0.9331744, -1.8260159, 1.8365273
2: -0.7276485, 1.1981939, -0.9160359, 1.2279017, -1.9555502, 2.1142297
3: -0.9435567, 0.8207595, -1.0935287, 0.8823293, -1.8258860, 1.9142883
4: -1.1971776, 0.9476389, -1.3500848, 1.0339899, -2.2311676, 2.2977238
5: -0.9528845, 0.9844075, -1.0665252, 1.0674635, -2.0203481, 2.0509326
6: -0.9440743, 0.9422595, -1.0561177, 1.0272465, -1.9713209, 1.9983771
7: -1.0863340, 1.0054618, -1.2233033, 1.1003927, -2.1867266, 2.2287650
8: -1.1422608, 1.0249797, -1.2902727, 1.1081519, -2.2504127, 2.3152523
9: -0.9996936, 1.1036472, -1.1070288, 1.2204630, -2.2201567, 2.2106762

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
time: 3.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
time: 2.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.7919588, 2.9467027, -1.1628827, 1.0031489, -4.7951078, 4.1095853
1: -2.9406590, 2.7310686, -0.9822426, 0.9331744, -3.8738334, 3.7133112
2: -4.8264527, 2.1360939, -0.9160359, 1.2279017, -6.0543547, 3.0521297
3: -4.2560358, 2.2299223, -1.0935287, 0.8823293, -5.1383653, 3.3234510
4: -4.5482984, 2.9002113, -1.3500848, 1.0339899, -5.5822883, 4.2502961
5: -3.4544127, 3.0389261, -1.0665252, 1.0674635, -4.5218763, 4.1054516
6: -3.6136959, 3.0776973, -1.0561177, 1.0272465, -4.6409426, 4.1338148
7: -4.1456871, 3.1120200, -1.2233033, 1.1003927, -5.2460799, 4.3353233
8: -4.7225423, 2.8873305, -1.2902727, 1.1081519, -5.8306942, 4.1776032
9: -3.4062939, 3.8535600, -1.1070288, 1.2204630, -4.6267567, 4.9605889

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
time: 3.77 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
time: 3.93 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.0531604, 0.9272488, -1.4929535, 1.2450616, -2.2982221, 2.4202023
1: -0.9028684, 0.8625062, -1.2301675, 1.1522005, -2.0550690, 2.0926738
2: -0.7486366, 1.2010515, -1.4371502, 1.3172511, -2.0658877, 2.6382017
3: -0.9602711, 0.8273244, -1.4943238, 1.0521894, -2.0124605, 2.3216481
4: -1.2139385, 0.9570729, -1.7495033, 1.2649043, -2.4788427, 2.7065761
5: -0.9654055, 0.9935039, -1.3713409, 1.2955949, -2.2610004, 2.3648448
6: -0.9571872, 0.9516574, -1.3735831, 1.2727892, -2.2299764, 2.3252406
7: -1.1019704, 1.0151452, -1.5921925, 1.3525438, -2.4545143, 2.6073377
8: -1.1584860, 1.0340253, -1.7415417, 1.3267570, -2.4852428, 2.7755671
9: -1.0113906, 1.1161186, -1.3987327, 1.5542634, -2.5656538, 2.5148511

Time for backsubstitution: 1.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
time: 3.94 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
time: 3.05 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.8652687, 3.0007892, -1.4929535, 1.2450616, -5.1103306, 4.4937429
1: -2.9992514, 2.7792504, -1.2301675, 1.1522005, -4.1514521, 4.0094180
2: -4.9265528, 2.1662290, -1.4371502, 1.3172511, -6.2438040, 3.6033792
3: -4.3435698, 2.2665625, -1.4943238, 1.0521894, -5.3957591, 3.7608862
4: -4.6355057, 2.9509566, -1.7495033, 1.2649043, -5.9004097, 4.7004600
5: -3.5174999, 3.0933805, -1.3713409, 1.2955949, -4.8130951, 4.4647212
6: -3.6814132, 3.1353431, -1.3735831, 1.2727892, -4.9542027, 4.5089264
7: -4.2242031, 3.1676087, -1.5921925, 1.3525438, -5.5767469, 4.7598014
8: -4.8138123, 2.9366531, -1.7415417, 1.3267570, -6.1405692, 4.6781950
9: -3.4669244, 3.9269178, -1.3987327, 1.5542634, -5.0211878, 5.3256502

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313079
time: 2.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313078
time: 3.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.4499955, 1.2108288, -1.3771781, 1.1589034, -2.6088989, 2.5880070
1: -1.1970177, 1.1183782, -1.1416976, 1.0740740, -2.2710917, 2.2600758
2: -1.3697586, 1.3010416, -1.2538208, 1.2846625, -2.6544211, 2.5548625
3: -1.4400407, 1.0272079, -1.3523948, 0.9921165, -2.4321570, 2.3796027
4: -1.6930354, 1.2307545, -1.6088462, 1.1824472, -2.8754826, 2.8396006
5: -1.3290328, 1.2621032, -1.2636000, 1.2135365, -2.5425692, 2.5257032
6: -1.3356463, 1.2411077, -1.2616413, 1.1848114, -2.5204577, 2.5027490
7: -1.5448251, 1.3125018, -1.4614695, 1.2636461, -2.8084712, 2.7739713
8: -1.6804488, 1.2949226, -1.5828717, 1.2480716, -2.9285202, 2.8777943
9: -1.3553064, 1.5072877, -1.2941175, 1.4359045, -2.7912109, 2.8014052

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281545
time: 3.70 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
time: 4.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.8693132, 3.7401671, -1.3771781, 1.1589034, -6.0282164, 5.1173453
1: -3.8040197, 3.4359655, -1.1416976, 1.0740740, -4.8780937, 4.5776634
2: -6.3038125, 2.6092558, -1.2538208, 1.2846625, -7.5884752, 3.8630767
3: -5.5297933, 2.7702694, -1.3523948, 0.9921165, -6.5219097, 4.1226645
4: -5.8126616, 3.6458583, -1.6088462, 1.1824472, -6.9951086, 5.2547045
5: -4.3859935, 3.8539329, -1.2636000, 1.2135365, -5.5995302, 5.1175327
6: -4.6104403, 3.9202943, -1.2616413, 1.1848114, -5.7952518, 5.1819353
7: -5.3008614, 3.9267373, -1.4614695, 1.2636461, -6.5645075, 5.3882070
8: -6.0561352, 3.6161757, -1.5828717, 1.2480716, -7.3042068, 5.1990471
9: -4.2908692, 4.9178047, -1.2941175, 1.4359045, -5.7267737, 6.2119222

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
time: 2.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4418779, upper bound: 7.4237173
time: 4.65 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.5572910, 1.2917411, -1.7683985, 1.4520612, -3.0093522, 3.0601397
1: -1.2789603, 1.1921618, -1.4402798, 1.3452897, -2.6242499, 2.6324415
2: -1.5395666, 1.3316808, -1.8708315, 1.3969896, -2.9365563, 3.2025123
3: -1.5709763, 1.0829575, -1.8287210, 1.1967047, -2.7676811, 2.9116786
4: -1.8234650, 1.3076975, -2.0820878, 1.4649547, -3.2884197, 3.3897853
5: -1.4298999, 1.3392417, -1.6293851, 1.4950441, -2.9249439, 2.9686270
6: -1.4390669, 1.3236289, -1.6457657, 1.4859622, -2.9250290, 2.9693947
7: -1.6675868, 1.3947434, -1.9053166, 1.5667646, -3.2343514, 3.3000600
8: -1.8282120, 1.3676143, -2.1217227, 1.5141705, -3.3423824, 3.4893370
9: -1.4533160, 1.6178205, -1.6505555, 1.8376806, -3.2909966, 3.2683759

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
time: 2.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
time: 4.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.0666490, 3.8846729, -1.7683985, 1.4520612, -6.5187101, 5.6530714
1: -3.9632299, 3.5649397, -1.4402798, 1.3452897, -5.3085194, 5.0052195
2: -6.5705166, 2.6902735, -1.8708315, 1.3969896, -7.9675064, 4.5611048
3: -5.7645931, 2.8702362, -1.8287210, 1.1967047, -6.9612980, 4.6989574
4: -6.0453620, 3.7824497, -2.0820878, 1.4649547, -7.5103168, 5.8645372
5: -4.5549335, 4.0001926, -1.6293851, 1.4950441, -6.0499778, 5.6295776
6: -4.7944660, 4.0743966, -1.6457657, 1.4859622, -6.2804279, 5.7201624
7: -5.5124435, 4.0756145, -1.9053166, 1.5667646, -7.0792084, 5.9809313
8: -6.3009572, 3.7499282, -2.1217227, 1.5141705, -7.8151278, 5.8716507
9: -4.4531837, 5.1136880, -1.6505555, 1.8376806, -6.2908640, 6.7642436

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849
time: 2.44 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4419384, upper bound: 7.4419384
time: 4.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.84 seconds
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6283250
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5459817, upper bound: 7.6311681
NS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5459817, upper bound: 7.4237516
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5517384, upper bound: 7.6289078
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282343
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6282341
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5391198, upper bound: 7.4237475
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5391198, upper bound: 7.6313000
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6288619
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.5454093, upper bound: 7.6315043
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6282668
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4687658, upper bound: 7.6311510
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6288433
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313079
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4714717, upper bound: 7.6313078
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281545
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6281547
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4418779, upper bound: 7.6312818
NS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4418779, upper bound: 7.4237173
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6287830
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4419384, upper bound: 7.6314849
NS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.84
Output dim: 2, lower bound: -7.4419384, upper bound: 7.4419384

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0999603, 0.1069386, -0.8551247, 0.7983816, -0.8983418, 0.9620633
1: -0.1372842, 0.1415909, -0.7615308, 0.7482750, -0.8855592, 0.9031218
2: 0.7027340, 1.0396814, -0.4506930, 1.1650431, -0.4623091, 1.4903744
3: -0.0289697, 0.2068110, -0.7235693, 0.7370088, -0.7659785, 0.9303803
4: -0.1513371, 0.1660503, -0.9802259, 0.8269940, -0.9783311, 1.1462762
5: -0.1289959, 0.1403148, -0.7923279, 0.8684083, -0.9974041, 0.9326428
6: -0.1125522, 0.1390101, -0.7721305, 0.8151376, -0.9276898, 0.9111406
7: -0.1393690, 0.1629245, -0.8798913, 0.8797811, -1.0191501, 1.0428158
8: -0.1359288, 0.2251918, -0.9418252, 0.9057338, -1.0416626, 1.1670170
9: -0.1500749, 0.1279543, -0.8458427, 0.9452844, -1.0953593, 0.9737970

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710082
time: 3.61 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
time: 3.87 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0999603, 0.1069386, -3.6476877, 2.7839622, -2.8839226, 3.7546263
1: -0.1372842, 0.1415909, -2.8018668, 2.5745392, -2.7118235, 2.9434576
2: 0.7027340, 1.0396814, -4.8289142, 1.8660736, -1.1633396, 5.8685956
3: -0.0289697, 0.2068110, -4.0722418, 2.1583340, -2.1873038, 4.2790527
4: -0.1513371, 0.1660503, -4.3325849, 2.7284336, -2.8797708, 4.4986353
5: -0.1289959, 0.1403148, -3.3557072, 2.7416050, -2.8706009, 3.4960220
6: -0.1125522, 0.1390101, -3.4450753, 2.8407545, -2.9533067, 3.5840852
7: -0.1393690, 0.1629245, -3.9689660, 2.9865489, -3.1259179, 4.1318903
8: -0.1359288, 0.2251918, -4.6700106, 2.7224474, -2.8583763, 4.8952022
9: -0.1500749, 0.1279543, -3.2760303, 3.6902027, -3.8402777, 3.4039845

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710083
time: 3.63 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
time: 2.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.5137417, 0.5817116, -0.8551247, 0.7983816, -1.3121233, 1.4368362
1: -0.5447940, 0.5669526, -0.7615308, 0.7482750, -1.2930690, 1.3284833
2: 0.0311265, 1.1043226, -0.4506930, 1.1650431, -1.1339166, 1.5550156
3: -0.3708899, 0.5945812, -0.7235693, 0.7370088, -1.1078987, 1.3181505
4: -0.5962261, 0.6047728, -0.9802259, 0.8269940, -1.4232202, 1.5849987
5: -0.5363327, 0.6336310, -0.7923279, 0.8684083, -1.4047409, 1.4259589
6: -0.4864335, 0.6049811, -0.7721305, 0.8151376, -1.3015711, 1.3771117
7: -0.5493956, 0.6686618, -0.8798913, 0.8797811, -1.4291768, 1.5485532
8: -0.6184307, 0.7035065, -0.9418252, 0.9057338, -1.5241646, 1.6453317
9: -0.5875289, 0.6464775, -0.8458427, 0.9452844, -1.5328133, 1.4923202

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292519, upper bound: 7.5765304
time: 4.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6065362, upper bound: 7.5755026
time: 4.11 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0997177, 0.1066720, -1.1652219, 1.0053251, -1.1050427, 1.2718939
1: -0.1370031, 0.1412840, -0.9838070, 0.9347872, -1.0717902, 1.1250910
2: 0.7031748, 1.0396798, -0.9202113, 1.2288338, -0.5256590, 1.9598911
3: -0.0287646, 0.2064803, -1.0969396, 0.8832272, -0.9119918, 1.3034199
4: -0.1510483, 0.1658331, -1.3542893, 1.0352956, -1.1863439, 1.5201224
5: -0.1287037, 0.1399902, -1.0680280, 1.0702395, -1.1989433, 1.2080182
6: -0.1123485, 0.1386405, -1.0581671, 1.0293008, -1.1416494, 1.1968076
7: -0.1391573, 0.1625858, -1.2263916, 1.1025243, -1.2416816, 1.3889774
8: -0.1355999, 0.2248350, -1.2947221, 1.1090962, -1.2446960, 1.5195570
9: -0.1497552, 0.1276529, -1.1098964, 1.2230521, -1.3728074, 1.2375493

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
time: 3.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
time: 2.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0997177, 0.1066720, -4.0253606, 3.1445298, -3.2442474, 4.1320324
1: -0.1370031, 0.1412840, -3.1651258, 2.9174316, -3.0544348, 3.3064098
2: 0.7031748, 1.0396798, -5.4192858, 2.0534806, -1.3503058, 6.4589653
3: -0.0287646, 0.2064803, -4.5677948, 2.3638389, -2.3926036, 4.7742753
4: -0.1510483, 0.1658331, -4.8053541, 3.0876324, -3.2386806, 4.9711871
5: -0.1287037, 0.1399902, -3.7480755, 3.1195178, -3.2482216, 3.8880658
6: -0.1123485, 0.1386405, -3.8025064, 3.2269757, -3.3393242, 3.9411469
7: -0.1391573, 0.1625858, -4.4761119, 3.2936349, -3.4327922, 4.6386976
8: -0.1355999, 0.2248350, -5.2307100, 3.0486403, -3.1842401, 5.4555449
9: -0.1497552, 0.1276529, -3.7011733, 4.1598730, -4.3096280, 3.8288262

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
time: 3.17 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
time: 2.18 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.1481730, 0.1599290, -1.0619552, 0.9352350, -1.0834080, 1.2218843
1: -0.1931481, 0.2025903, -0.9091503, 0.8703966, -1.0635448, 1.1117406
2: 0.6151205, 1.0425050, -0.7625545, 1.2056801, -0.5905596, 1.8050596
3: -0.0697279, 0.2725243, -0.9726491, 0.8332772, -0.9030051, 1.2451735
4: -0.2087196, 0.2092075, -1.2301250, 0.9645168, -1.1732364, 1.4393325
5: -0.1870432, 0.2048456, -0.9752898, 1.0026104, -1.1896536, 1.1801354
6: -0.1530457, 0.2124394, -0.9618854, 0.9570676, -1.1101133, 1.1743248
7: -0.1814390, 0.2302229, -1.1108429, 1.0267212, -1.2081603, 1.3410658
8: -0.2012803, 0.2961308, -1.1707318, 1.0411508, -1.2424310, 1.4668627
9: -0.2136270, 0.1878617, -1.0229536, 1.1269178, -1.3405448, 1.2108153

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516038
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
time: 7.13 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.1481730, 0.1599290, -3.8284457, 2.9689906, -3.1171637, 3.9883747
1: -0.1931481, 0.2025903, -3.0065353, 2.7203574, -2.9135056, 3.2091255
2: 0.6151205, 1.0425050, -5.1298418, 1.9725821, -1.3574616, 6.1723471
3: -0.0697279, 0.2725243, -4.3575759, 2.2571936, -2.3269215, 4.6301003
4: -0.2087196, 0.2092075, -4.5779529, 2.9183955, -3.1271150, 4.7871604
5: -0.1870432, 0.2048456, -3.5277989, 2.9429975, -3.1300406, 3.7326446
6: -0.1530457, 0.2124394, -3.6316020, 3.0413005, -3.1943462, 3.8440413
7: -0.1814390, 0.2302229, -4.1981115, 3.1519873, -3.3334262, 4.4283342
8: -0.2012803, 0.2961308, -4.9361658, 2.9024711, -3.1037514, 5.2322965
9: -0.2136270, 0.1878617, -3.4911053, 3.9163158, -4.1299429, 3.6789670

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516039
time: 2.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
time: 2.19 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.3306952, 1.1253389, -3.8284457, 2.9689906, -4.2996855, 4.9537845
1: -1.1061755, 1.0433774, -3.0065353, 2.7203574, -3.8265328, 4.0499125
2: -1.1810069, 1.2723873, -5.1298418, 1.9725821, -3.1535890, 6.4022293
3: -1.2967119, 0.9677966, -4.3575759, 2.2571936, -3.5539055, 5.3253727
4: -1.5545385, 1.1491244, -4.5779529, 2.9183955, -4.4729338, 5.7270775
5: -1.2197224, 1.1831168, -3.5277989, 2.9429975, -4.1627197, 4.7109156
6: -1.2164470, 1.1506547, -3.6316020, 3.0413005, -4.2577477, 4.7822566
7: -1.4102179, 1.2283047, -4.1981115, 3.1519873, -4.5622053, 5.4264164
8: -1.5208342, 1.2165208, -4.9361658, 2.9024711, -4.4233055, 6.1526866
9: -1.2542810, 1.3892401, -3.4911053, 3.9163158, -5.1705971, 4.8803453

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6298158, upper bound: 7.5768564
time: 5.34 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279708, upper bound: 7.5767800
time: 17.35 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.1587485, 0.1715083, -1.4410763, 1.2068056, -1.3655541, 1.6125846
1: -0.2053554, 0.2160324, -1.1903447, 1.1171834, -1.3225389, 1.4063771
2: 0.5959755, 1.0424958, -1.3553936, 1.3030133, -0.7070378, 2.3978896
3: -0.0786342, 0.2869375, -1.4313635, 1.0249015, -1.1035357, 1.7183011
4: -0.2215261, 0.2186382, -1.6877695, 1.2274487, -1.4489748, 1.9064077
5: -0.1997317, 0.2189467, -1.3219635, 1.2599572, -1.4596889, 1.5409102
6: -0.1618941, 0.2286291, -1.3230343, 1.2336626, -1.3955567, 1.5516634
7: -0.1906322, 0.2452627, -1.5333601, 1.3132799, -1.5039121, 1.7786229
8: -0.2155851, 0.3116325, -1.6715739, 1.2907040, -1.5062891, 1.9832064
9: -0.2276510, 0.2009524, -1.3525006, 1.5009515, -1.7286025, 1.5534530

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
time: 3.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
time: 2.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1587485, 0.1715083, -4.3890800, 3.3949044, -3.5536530, 4.5605884
1: -0.2053554, 0.2160324, -3.4329352, 3.1820660, -3.3874214, 3.6489677
2: 0.5959755, 1.0424958, -5.9047050, 2.1749189, -1.5789434, 6.9472008
3: -0.0786342, 0.2869375, -4.9883642, 2.5989501, -2.6775844, 5.2753019
4: -0.2215261, 0.2186382, -5.1978326, 3.3766735, -3.5981996, 5.4164705
5: -0.1997317, 0.2189467, -4.0887527, 3.3756824, -3.5754142, 4.3076992
6: -0.1618941, 0.2286291, -4.3802404, 3.5210381, -3.6829321, 4.6088696
7: -0.1906322, 0.2452627, -4.8307076, 3.6130788, -3.8037109, 5.0759702
8: -0.2155851, 0.3116325, -5.7200208, 3.2789156, -3.4945006, 6.0316534
9: -0.2276510, 0.2009524, -4.0253072, 4.5143766, -4.7420278, 4.2262597

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
time: 2.52 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
time: 2.54 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.4588451, 1.2203939, -1.4410763, 1.2068056, -2.6656508, 2.6614702
1: -1.2039182, 1.1288085, -1.1903447, 1.1171834, -2.3211017, 2.3191533
2: -1.3840015, 1.3078859, -1.3553936, 1.3030133, -2.6870148, 2.6632795
3: -1.4534780, 1.0339497, -1.4313635, 1.0249015, -2.4783795, 2.4653132
4: -1.7099203, 1.2396287, -1.6877695, 1.2274487, -2.9373689, 2.9273982
5: -1.3381892, 1.2731776, -1.3219635, 1.2599572, -2.5981464, 2.5951412
6: -1.3403305, 1.2472912, -1.3230343, 1.2336626, -2.5739932, 2.5703254
7: -1.5539688, 1.3266928, -1.5333601, 1.3132799, -2.8672485, 2.8600531
8: -1.6963817, 1.3024652, -1.6715739, 1.2907040, -2.9870858, 2.9740391
9: -1.3689091, 1.5193797, -1.3525006, 1.5009515, -2.8698606, 2.8718803

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255394
time: 3.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254628
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.4588451, 1.2203939, -4.3890800, 3.3949044, -4.8537493, 5.6094742
1: -1.2039182, 1.1288085, -3.4329352, 3.1820660, -4.3859844, 4.5617437
2: -1.3840015, 1.3078859, -5.9047050, 2.1749189, -3.5589204, 7.2125912
3: -1.4534780, 1.0339497, -4.9883642, 2.5989501, -4.0524282, 6.0223141
4: -1.7099203, 1.2396287, -5.1978326, 3.3766735, -5.0865936, 6.4374614
5: -1.3381892, 1.2731776, -4.0887527, 3.3756824, -4.7138715, 5.3619304
6: -1.3403305, 1.2472912, -4.3802404, 3.5210381, -4.8613687, 5.6275315
7: -1.5539688, 1.3266928, -4.8307076, 3.6130788, -5.1670475, 6.1574001
8: -1.6963817, 1.3024652, -5.7200208, 3.2789156, -4.9752975, 7.0224857
9: -1.3689091, 1.5193797, -4.0253072, 4.5143766, -5.8832855, 5.5446868

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255393
time: 3.28 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254617
time: 2.43 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.0393264, 0.9178565, -0.8551247, 0.7983816, -1.8377080, 1.7729812
1: -0.8928415, 0.8542848, -0.7615308, 0.7482750, -1.6411166, 1.6158156
2: -0.7276485, 1.1981939, -0.4506930, 1.1650431, -1.8926916, 1.6488869
3: -0.9435567, 0.8207595, -0.7235693, 0.7370088, -1.6805655, 1.5443289
4: -1.1971776, 0.9476389, -0.9802259, 0.8269940, -2.0241716, 1.9278648
5: -0.9528845, 0.9844075, -0.7923279, 0.8684083, -1.8212928, 1.7767354
6: -0.9440743, 0.9422595, -0.7721305, 0.8151376, -1.7592120, 1.7143900
7: -1.0863340, 1.0054618, -0.8798913, 0.8797811, -1.9661151, 1.8853531
8: -1.1422608, 1.0249797, -0.9418252, 0.9057338, -2.0479946, 1.9668050
9: -0.9996936, 1.1036472, -0.8458427, 0.9452844, -1.9449780, 1.9494900

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687311
time: 3.79 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
time: 2.12 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.0393264, 0.9178565, -3.6216893, 2.7327540, -3.7720804, 4.5395460
1: -0.8928415, 0.8542848, -2.7621019, 2.5255127, -3.4183543, 3.6163867
2: -0.7276485, 1.1981939, -4.7444296, 1.8587983, -2.5864468, 5.9426236
3: -0.9435567, 0.8207595, -4.0338159, 2.1043594, -3.0479159, 4.8545752
4: -1.1971776, 0.9476389, -4.3264675, 2.7139330, -3.9111106, 5.2741065
5: -0.9528845, 0.9844075, -3.2967334, 2.6955271, -3.6484115, 4.2811408
6: -0.9440743, 0.9422595, -3.3324790, 2.8237011, -3.7677755, 4.2747383
7: -1.0863340, 1.0054618, -3.9608104, 2.9322450, -4.0185790, 4.9662724
8: -1.1422608, 1.0249797, -4.5993862, 2.7224474, -3.8647082, 5.6243658
9: -0.9996936, 1.1036472, -3.1571589, 3.6648762, -4.6645699, 4.2608061

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687310
time: 4.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
time: 3.49 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.7919588, 2.9467027, -0.8551247, 0.7983816, -4.5903406, 3.8018274
1: -2.9406590, 2.7310686, -0.7615308, 0.7482750, -3.6889341, 3.4925995
2: -4.8264527, 2.1360939, -0.4506930, 1.1650431, -5.9914961, 2.5867867
3: -4.2560358, 2.2299223, -0.7235693, 0.7370088, -4.9930449, 2.9534917
4: -4.5482984, 2.9002113, -0.9802259, 0.8269940, -5.3752923, 3.8804374
5: -3.4544127, 3.0389261, -0.7923279, 0.8684083, -4.3228211, 3.8312540
6: -3.6136959, 3.0776973, -0.7721305, 0.8151376, -4.4288335, 3.8498278
7: -4.1456871, 3.1120200, -0.8798913, 0.8797811, -5.0254683, 3.9919114
8: -4.7225423, 2.8873305, -0.9418252, 0.9057338, -5.6282759, 3.8291557
9: -3.4062939, 3.8535600, -0.8458427, 0.9452844, -4.3515782, 4.6994028

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
time: 2.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
time: 2.56 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.7919588, 2.9467027, -3.6216893, 2.7327540, -6.5247126, 6.5683918
1: -2.9406590, 2.7310686, -2.7621019, 2.5255127, -5.4661717, 5.4931707
2: -4.8264527, 2.1360939, -4.7444296, 1.8587983, -6.6852512, 6.8805237
3: -4.2560358, 2.2299223, -4.0338159, 2.1043594, -6.3603954, 6.2637382
4: -4.5482984, 2.9002113, -4.3264675, 2.7139330, -7.2622313, 7.2266788
5: -3.4544127, 3.0389261, -3.2967334, 2.6955271, -6.1499395, 6.3356595
6: -3.6136959, 3.0776973, -3.3324790, 2.8237011, -6.4373970, 6.4101763
7: -4.1456871, 3.1120200, -3.9608104, 2.9322450, -7.0779324, 7.0728302
8: -4.7225423, 2.8873305, -4.5993862, 2.7224474, -7.4449897, 7.4867167
9: -3.4062939, 3.8535600, -3.1571589, 3.6648762, -7.0711699, 7.0107188

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
time: 3.19 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
time: 3.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.0531604, 0.9272488, -1.1652219, 1.0053251, -2.0584855, 2.0924706
1: -0.9028684, 0.8625062, -0.9838070, 0.9347872, -1.8376555, 1.8463132
2: -0.7486366, 1.2010515, -0.9202113, 1.2288338, -1.9774704, 2.1212628
3: -0.9602711, 0.8273244, -1.0969396, 0.8832272, -1.8434983, 1.9242640
4: -1.2139385, 0.9570729, -1.3542893, 1.0352956, -2.2492342, 2.3113623
5: -0.9654055, 0.9935039, -1.0680280, 1.0702395, -2.0356450, 2.0615320
6: -0.9571872, 0.9516574, -1.0581671, 1.0293008, -1.9864880, 2.0098245
7: -1.1019704, 1.0151452, -1.2263916, 1.1025243, -2.2044947, 2.2415366
8: -1.1584860, 1.0340253, -1.2947221, 1.1090962, -2.2675822, 2.3287473
9: -1.0113906, 1.1161186, -1.1098964, 1.2230521, -2.2344427, 2.2260151

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
time: 3.24 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
time: 2.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.0531604, 0.9272488, -3.9990139, 3.1387386, -4.1918993, 4.9262629
1: -0.9028684, 0.8625062, -3.1564875, 2.8728855, -3.7757540, 4.0189939
2: -0.7486366, 1.2010515, -5.3902931, 2.0323970, -2.7810335, 6.5913448
3: -0.9602711, 0.8273244, -4.5473056, 2.3561730, -3.3164442, 5.3746300
4: -1.2139385, 0.9570729, -4.7793722, 3.0876324, -4.3015709, 5.7364450
5: -0.9654055, 0.9935039, -3.7325683, 3.0704665, -4.0358720, 4.7260723
6: -0.9571872, 0.9516574, -3.7873683, 3.1964283, -4.1536155, 4.7390256
7: -1.1019704, 1.0151452, -4.4685564, 3.2716124, -4.3735828, 5.4837017
8: -1.1584860, 1.0340253, -5.1899042, 3.0368407, -4.1953268, 6.2239294
9: -1.0113906, 1.1161186, -3.7011733, 4.1113167, -5.1227074, 4.8172917

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
time: 3.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
time: 6.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.8652687, 3.0007892, -1.1652219, 1.0053251, -4.8705940, 4.1660109
1: -2.9992514, 2.7792504, -0.9838070, 0.9347872, -3.9340386, 3.7630572
2: -4.9265528, 2.1662290, -0.9202113, 1.2288338, -6.1553864, 3.0864403
3: -4.3435698, 2.2665625, -1.0969396, 0.8832272, -5.2267971, 3.3635020
4: -4.6355057, 2.9509566, -1.3542893, 1.0352956, -5.6708012, 4.3052459
5: -3.5174999, 3.0933805, -1.0680280, 1.0702395, -4.5877395, 4.1614084
6: -3.6814132, 3.1353431, -1.0581671, 1.0293008, -4.7107139, 4.1935101
7: -4.2242031, 3.1676087, -1.2263916, 1.1025243, -5.3267274, 4.3940001
8: -4.8138123, 2.9366531, -1.2947221, 1.1090962, -5.9229083, 4.2313752
9: -3.4669244, 3.9269178, -1.1098964, 1.2230521, -4.6899767, 5.0368142

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289030, upper bound: 7.6223652
time: 3.16 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5621397, upper bound: 7.6201651
time: 3.14 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.8652687, 3.0007892, -3.9990139, 3.1387386, -7.0040073, 6.9998031
1: -2.9992514, 2.7792504, -3.1564875, 2.8728855, -5.8721371, 5.9357376
2: -4.9265528, 2.1662290, -5.3902931, 2.0323970, -6.9589500, 7.5565224
3: -4.3435698, 2.2665625, -4.5473056, 2.3561730, -6.6997428, 6.8138680
4: -4.6355057, 2.9509566, -4.7793722, 3.0876324, -7.7231379, 7.7303286
5: -3.5174999, 3.0933805, -3.7325683, 3.0704665, -6.5879664, 6.8259487
6: -3.6814132, 3.1353431, -3.7873683, 3.1964283, -6.8778415, 6.9227114
7: -4.2242031, 3.1676087, -4.4685564, 3.2716124, -7.4958153, 7.6361651
8: -4.8138123, 2.9366531, -5.1899042, 3.0368407, -7.8506527, 8.1265574
9: -3.4669244, 3.9269178, -3.7011733, 4.1113167, -7.5782413, 7.6280909

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289030, upper bound: 7.6223648
time: 3.09 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5621397, upper bound: 7.6201652
time: 2.36 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.4499955, 1.2108288, -1.0619552, 0.9352350, -2.3852305, 2.2727840
1: -1.1970177, 1.1183782, -0.9091503, 0.8703966, -2.0674143, 2.0275285
2: -1.3697586, 1.3010416, -0.7625545, 1.2056801, -2.5754387, 2.0635962
3: -1.4400407, 1.0272079, -0.9726491, 0.8332772, -2.2733178, 1.9998569
4: -1.6930354, 1.2307545, -1.2301250, 0.9645168, -2.6575522, 2.4608793
5: -1.3290328, 1.2621032, -0.9752898, 1.0026104, -2.3316431, 2.2373929
6: -1.3356463, 1.2411077, -0.9618854, 0.9570676, -2.2927139, 2.2029932
7: -1.5448251, 1.3125018, -1.1108429, 1.0267212, -2.5715463, 2.4233446
8: -1.6804488, 1.2949226, -1.1707318, 1.0411508, -2.7215996, 2.4656544
9: -1.3553064, 1.5072877, -1.0229536, 1.1269178, -2.4822242, 2.5302415

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2398641, upper bound: 7.2486002
time: 2.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0286784, upper bound: 7.2183160
time: 7.26 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.4499955, 1.2108288, -3.8284457, 2.9689906, -4.4189863, 5.0392742
1: -1.1970177, 1.1183782, -3.0065353, 2.7203574, -3.9173751, 4.1249132
2: -1.3697586, 1.3010416, -5.1298418, 1.9725821, -3.3423407, 6.4308834
3: -1.4400407, 1.0272079, -4.3575759, 2.2571936, -3.6972342, 5.3847837
4: -1.6930354, 1.2307545, -4.5779529, 2.9183955, -4.6114311, 5.8087072
5: -1.3290328, 1.2621032, -3.5277989, 2.9429975, -4.2720304, 4.7899022
6: -1.3356463, 1.2411077, -3.6316020, 3.0413005, -4.3769469, 4.8727098
7: -1.5448251, 1.3125018, -4.1981115, 3.1519873, -4.6968126, 5.5106134
8: -1.6804488, 1.2949226, -4.9361658, 2.9024711, -4.5829201, 6.2310886
9: -1.3553064, 1.5072877, -3.4911053, 3.9163158, -5.2716222, 4.9983931

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2398641, upper bound: 7.2486002
time: 4.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0286784, upper bound: 7.2183160
time: 3.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -4.8693132, 3.7401671, -1.0619552, 0.9352350, -5.8045483, 4.8021221
1: -3.8040197, 3.4359655, -0.9091503, 0.8703966, -4.6744165, 4.3451157
2: -6.3038125, 2.6092558, -0.7625545, 1.2056801, -7.5094929, 3.3718104
3: -5.5297933, 2.7702694, -0.9726491, 0.8332772, -6.3630705, 3.7429185
4: -5.8126616, 3.6458583, -1.2301250, 0.9645168, -6.7771783, 4.8759832
5: -4.3859935, 3.8539329, -0.9752898, 1.0026104, -5.3886042, 4.8292227
6: -4.6104403, 3.9202943, -0.9618854, 0.9570676, -5.5675077, 4.8821797
7: -5.3008614, 3.9267373, -1.1108429, 1.0267212, -6.3275824, 5.0375805
8: -6.0561352, 3.6161757, -1.1707318, 1.0411508, -7.0972862, 4.7869072
9: -4.2908692, 4.9178047, -1.0229536, 1.1269178, -5.4177871, 5.9407582

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6294207, upper bound: 7.5768333
time: 2.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6251355, upper bound: 7.5767682
time: 2.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.5572910, 1.2917411, -1.4410763, 1.2068056, -2.7640967, 2.7328174
1: -1.2789603, 1.1921618, -1.1903447, 1.1171834, -2.3961439, 2.3825064
2: -1.5395666, 1.3316808, -1.3553936, 1.3030133, -2.8425798, 2.6870744
3: -1.5709763, 1.0829575, -1.4313635, 1.0249015, -2.5958776, 2.5143209
4: -1.8234650, 1.3076975, -1.6877695, 1.2274487, -3.0509138, 2.9954672
5: -1.4298999, 1.3392417, -1.3219635, 1.2599572, -2.6898570, 2.6612053
6: -1.4390669, 1.3236289, -1.3230343, 1.2336626, -2.6727295, 2.6466632
7: -1.6675868, 1.3947434, -1.5333601, 1.3132799, -2.9808667, 2.9281034
8: -1.8282120, 1.3676143, -1.6715739, 1.2907040, -3.1189160, 3.0391881
9: -1.4533160, 1.6178205, -1.3525006, 1.5009515, -2.9542675, 2.9703212

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2474531, upper bound: 7.2814307
time: 5.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0287855, upper bound: 7.2567092
time: 3.67 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.5572910, 1.2917411, -4.3312969, 3.3843651, -4.9416561, 5.6230383
1: -1.2789603, 1.1921618, -3.3857274, 3.1820660, -4.4610262, 4.5778894
2: -1.5395666, 1.3316808, -5.9047050, 2.1405101, -3.6800766, 7.2363858
3: -1.5709763, 1.0829575, -4.9427772, 2.5798097, -4.1507859, 6.0257349
4: -1.8234650, 1.3076975, -5.1705179, 3.3620133, -5.1854782, 6.4782152
5: -1.4298999, 1.3392417, -4.0589714, 3.3540003, -4.7839003, 5.3982134
6: -1.4390669, 1.3236289, -4.3133488, 3.5210381, -4.9601049, 5.6369777
7: -1.6675868, 1.3947434, -4.7984152, 3.6130788, -5.2806654, 6.1931586
8: -1.8282120, 1.3676143, -5.6979742, 3.2687616, -5.0969734, 7.0655885
9: -1.4533160, 1.6178205, -3.9943962, 4.4870510, -5.9403667, 5.6122169

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2474531, upper bound: 7.2814307
time: 8.23 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0287855, upper bound: 7.2567092
time: 1.88 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -5.0666490, 3.8846729, -1.4410763, 1.2068056, -6.2734547, 5.3257494
1: -3.9632299, 3.5649397, -1.1903447, 1.1171834, -5.0804133, 4.7552843
2: -6.5705166, 2.6902735, -1.3553936, 1.3030133, -7.8735299, 4.0456672
3: -5.7645931, 2.8702362, -1.4313635, 1.0249015, -6.7894945, 4.3015995
4: -6.0453620, 3.7824497, -1.6877695, 1.2274487, -7.2728109, 5.4702191
5: -4.5549335, 4.0001926, -1.3219635, 1.2599572, -5.8148909, 5.3221560
6: -4.7944660, 4.0743966, -1.3230343, 1.2336626, -6.0281286, 5.3974309
7: -5.5124435, 4.0756145, -1.5333601, 1.3132799, -6.8257236, 5.6089745
8: -6.3009572, 3.7499282, -1.6715739, 1.2907040, -7.5916615, 5.4215021
9: -4.4531837, 5.1136880, -1.3525006, 1.5009515, -5.9541349, 6.4661884

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6296323, upper bound: 7.6255162
time: 5.95 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6254506, upper bound: 7.6254489
time: 2.09 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 8.90 seconds
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710082
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.3050140, upper bound: 7.2710083
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1434126, upper bound: 7.2490772
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6292519, upper bound: 7.5765304
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6065362, upper bound: 7.5755026
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.3103391, upper bound: 7.2959609
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1471496, upper bound: 7.2742580
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516038
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2866843, upper bound: 7.2516039
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0755991, upper bound: 7.2190859
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6298158, upper bound: 7.5768564
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6279708, upper bound: 7.5767800
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2940412, upper bound: 7.2844194
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0772492, upper bound: 7.2574310
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255394
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254628
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6299813, upper bound: 7.6255393
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6281657, upper bound: 7.6254617
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687311
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2582690, upper bound: 7.2687310
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1068228, upper bound: 7.2489089
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6287345, upper bound: 7.5764697
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.5620579, upper bound: 7.5754063
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2642471, upper bound: 7.2940383
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.1116652, upper bound: 7.2736696
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6289030, upper bound: 7.6223652
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.5621397, upper bound: 7.6201651
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6289030, upper bound: 7.6223648
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.5621397, upper bound: 7.6201652
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2398641, upper bound: 7.2486002
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0286784, upper bound: 7.2183160
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2398641, upper bound: 7.2486002
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0286784, upper bound: 7.2183160
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6294207, upper bound: 7.5768333
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6251355, upper bound: 7.5767682
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2474531, upper bound: 7.2814307
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0287855, upper bound: 7.2567092
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.2474531, upper bound: 7.2814307
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.0287855, upper bound: 7.2567092
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6296323, upper bound: 7.6255162
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 8.90
Output dim: 2, lower bound: -7.6254506, upper bound: 7.6254489

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.4343072, 0.5033215, -0.8436735, 0.7909445, -1.2252517, 1.3469950
1: -0.4735071, 0.5018988, -0.7535736, 0.7415912, -1.2150983, 1.2554724
2: 0.1553251, 1.0862610, -0.4334531, 1.1628922, -1.0075672, 1.5197141
3: -0.3106661, 0.5375999, -0.7099845, 0.7317411, -1.0424073, 1.2475843
4: -0.5082153, 0.5278337, -0.9664115, 0.8197320, -1.3279473, 1.4942452
5: -0.4655880, 0.5389364, -0.7824146, 0.8611367, -1.3267248, 1.3213511
6: -0.4091447, 0.5417210, -0.7619430, 0.8073134, -1.2164581, 1.3036640
7: -0.4661230, 0.5841292, -0.8673471, 0.8717926, -1.3379157, 1.4514762
8: -0.5304994, 0.6249281, -0.9295554, 0.8982729, -1.4287722, 1.5544834
9: -0.5185791, 0.5482144, -0.8360268, 0.9354415, -1.4540205, 1.3842412

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 58

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6196462, upper bound: 7.6274655
time: 2.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6196462, upper bound: 7.6274655
time: 3.69 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.18 + 597.57 = 603.75 seconds
