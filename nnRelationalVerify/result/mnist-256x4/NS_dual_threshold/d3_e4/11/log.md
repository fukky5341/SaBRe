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
execution time: IAR + RelationalAnalysis = 1.06 + 4.95 = 6.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6335958, upper bound: 7.6335962

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335479, upper bound: 7.6335761
time: 3.51 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335463, upper bound: 7.6335463
time: 2.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.06 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 6.06
Output dim: 2, lower bound: -7.6335479, upper bound: 7.6335761
NS_B2, status: Status.UNKNOWN, split count: 1, time: 6.06
Output dim: 2, lower bound: -7.6335463, upper bound: 7.6335463

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -4.7909756, 3.6853030, -3.7251990, 2.9015779, -7.6925535, 7.4105020
1: -3.7366765, 3.3927815, -2.8778124, 2.6893115, -6.4259882, 6.2705936
2: -6.2026405, 2.5792634, -4.7487893, 2.1088452, -8.3114853, 7.3280525
3: -5.4332085, 2.7331824, -4.1680532, 2.1933427, -7.6265512, 6.9012356
4: -5.7178774, 3.6008029, -4.4700050, 2.8505054, -8.5683823, 8.0708084
5: -4.3159161, 3.7948947, -3.3880157, 3.0042415, -7.3201575, 7.1829104
6: -4.5275164, 3.8596551, -3.5393488, 3.0220010, -7.5495176, 7.3990040
7: -5.2140265, 3.8749912, -4.0690737, 3.0618806, -8.2759075, 7.9440651
8: -5.9650707, 3.5711460, -4.6395154, 2.8396196, -8.8046904, 8.2106609
9: -4.2284045, 4.8461180, -3.3468199, 3.8004518, -8.0288563, 8.1929379

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 210

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330454, upper bound: 7.6330971
time: 2.40 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330765, upper bound: 7.6331121
time: 2.51 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -4.9511919, 3.8029757, -4.6887450, 3.6103084, -8.5615005, 8.4917202
1: -3.8656716, 3.4973142, -3.6534214, 3.3259871, -7.1916590, 7.1507359
2: -6.4220614, 2.6485746, -6.0639324, 2.5362964, -8.9583578, 8.7125072
3: -5.6225247, 2.8142078, -5.3117399, 2.6811905, -8.3037148, 8.1259480
4: -5.9051828, 3.7118700, -5.5978603, 3.5297210, -9.4349041, 9.3097305
5: -4.4537635, 3.9152968, -4.2277665, 3.7193160, -8.1730795, 8.1430635
6: -4.6768012, 3.9845791, -4.4315343, 3.7799363, -8.4567375, 8.4161129
7: -5.3861384, 3.9958732, -5.1041079, 3.7976844, -9.1838226, 9.0999813
8: -6.1632180, 3.6809516, -5.8385901, 3.5008440, -9.6640625, 9.5195417
9: -4.3596549, 5.0042391, -4.1445389, 4.7445335, -9.1041889, 9.1487780

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335463
time: 4.45 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335465
time: 1.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 7.26 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 7.26
Output dim: 2, lower bound: -7.6330454, upper bound: 7.6330971
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 7.26
Output dim: 2, lower bound: -7.6330765, upper bound: 7.6331121
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 7.26
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335463
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 7.26
Output dim: 2, lower bound: -7.6335464, upper bound: 7.6335465

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -2.9899712, 2.3628576, -2.9973176, 2.3697085, -5.3596797, 5.3601751
1: -2.3294446, 2.2065747, -2.3335009, 2.2069466, -4.5363913, 4.5400753
2: -3.7335980, 1.8199666, -3.7462654, 1.8130248, -5.5466228, 5.5662317
3: -3.2849290, 1.8319349, -3.2937617, 1.8342454, -5.1191745, 5.1256967
4: -3.5904775, 2.3491380, -3.6018212, 2.3500030, -5.9404802, 5.9509592
5: -2.7549639, 2.4502921, -2.7513871, 2.4577379, -5.2127018, 5.2016792
6: -2.8579345, 2.4540000, -2.8649039, 2.4561791, -5.3141136, 5.3189039
7: -3.2786298, 2.5131192, -3.2827492, 2.5135856, -5.7922153, 5.7958684
8: -3.7249141, 2.3525136, -3.7358642, 2.3512032, -6.0761175, 6.0883780
9: -2.7413840, 3.0623896, -2.7425733, 3.0774326, -5.8188167, 5.8049631

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6327716, upper bound: 7.6324390
time: 2.83 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323612, upper bound: 7.6324192
time: 3.88 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -3.4656997, 2.7083182, -3.1214218, 2.4605050, -5.9262047, 5.8297400
1: -2.6719747, 2.5218318, -2.4187989, 2.2900453, -4.9620199, 4.9406309
2: -4.3868742, 2.0041418, -3.9193032, 1.8571093, -6.2439833, 5.9234447
3: -3.8597226, 2.0659814, -3.4465160, 1.8956268, -5.7553492, 5.5124974
4: -4.1618671, 2.6753273, -3.7534587, 2.4359410, -6.5978079, 6.4287863
5: -3.1715245, 2.8048944, -2.8613255, 2.5510793, -5.7226038, 5.6662197
6: -3.3018813, 2.8223674, -2.9815106, 2.5530663, -5.8549476, 5.8038778
7: -3.7930665, 2.8692207, -3.4191084, 2.6067133, -6.3997798, 6.2883291
8: -4.3153620, 2.6711991, -3.8909020, 2.4353621, -6.7507238, 6.5621014
9: -3.1373191, 3.5353332, -2.8471899, 3.2023582, -6.3396773, 6.3825231

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6328218, upper bound: 7.6324557
time: 3.86 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324287, upper bound: 7.6324396
time: 2.54 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -3.7251990, 2.9015779, -4.6887450, 3.6103084, -7.3355074, 7.5903230
1: -2.8778124, 2.6893115, -3.6534214, 3.3259871, -6.2037992, 6.3427329
2: -4.7487893, 2.1088452, -6.0639324, 2.5362964, -7.2850857, 8.1727772
3: -4.1680532, 2.1933427, -5.3117399, 2.6811905, -6.8492436, 7.5050826
4: -4.4700050, 2.8505054, -5.5978603, 3.5297210, -7.9997263, 8.4483662
5: -3.3880157, 3.0042415, -4.2277665, 3.7193160, -7.1073318, 7.2320080
6: -3.5393488, 3.0220010, -4.4315343, 3.7799363, -7.3192854, 7.4535351
7: -4.0690737, 3.0618806, -5.1041079, 3.7976844, -7.8667583, 8.1659889
8: -4.6395154, 2.8396196, -5.8385901, 3.5008440, -8.1403599, 8.6782093
9: -3.3468199, 3.8004518, -4.1445389, 4.7445335, -8.0913534, 7.9449906

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330446
time: 7.65 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330748
time: 2.50 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -4.6887450, 3.6103084, -4.6887450, 3.6103084, -8.2990532, 8.2990532
1: -3.6534214, 3.3259871, -3.6534214, 3.3259871, -6.9794083, 6.9794083
2: -6.0639324, 2.5362964, -6.0639324, 2.5362964, -8.6002293, 8.6002293
3: -5.3117399, 2.6811905, -5.3117399, 2.6811905, -7.9929304, 7.9929304
4: -5.5978603, 3.5297210, -5.5978603, 3.5297210, -9.1275816, 9.1275816
5: -4.2277665, 3.7193160, -4.2277665, 3.7193160, -7.9470825, 7.9470825
6: -4.4315343, 3.7799363, -4.4315343, 3.7799363, -8.2114706, 8.2114706
7: -5.1041079, 3.7976844, -5.1041079, 3.7976844, -8.9017925, 8.9017925
8: -5.8385901, 3.5008440, -5.8385901, 3.5008440, -9.3394337, 9.3394337
9: -4.1445389, 4.7445335, -4.1445389, 4.7445335, -8.8890724, 8.8890724

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 210

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330441
time: 2.93 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330753
time: 2.86 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.69 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6327716, upper bound: 7.6324390
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6323612, upper bound: 7.6324192
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6328218, upper bound: 7.6324557
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6324287, upper bound: 7.6324396
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330446
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330748
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6330610, upper bound: 7.6330441
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 6.69
Output dim: 2, lower bound: -7.6330747, upper bound: 7.6330753

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -2.3151975, 1.8625641, -2.7276950, 2.1729178, -4.4881153, 4.5902591
1: -1.8558949, 1.7448099, -2.1480026, 2.0260634, -3.8819585, 3.8928125
2: -2.7631500, 1.5726111, -3.3686826, 1.7159586, -4.4791088, 4.9412937
3: -2.4932914, 1.4910572, -2.9669166, 1.7006347, -4.1939259, 4.4579735
4: -2.7511826, 1.8712901, -3.2738788, 2.1627779, -4.9139605, 5.1451688
5: -2.1427438, 1.9427977, -2.5115216, 2.2552068, -4.3979506, 4.4543190
6: -2.2092307, 1.9236181, -2.6118550, 2.2460961, -4.4553270, 4.5354729
7: -2.5191453, 2.0075314, -2.9860487, 2.3115568, -4.8307018, 4.9935799
8: -2.8667698, 1.8952205, -3.4003899, 2.1680882, -5.0348577, 5.2956104
9: -2.1585655, 2.3879640, -2.5151827, 2.8083096, -4.9668751, 4.9031467

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
time: 4.59 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
time: 2.55 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -5.7190075, 4.3689775, -2.6799695, 2.1383300, -7.8573375, 7.0489473
1: -4.2014623, 4.0458031, -2.1153226, 1.9939835, -6.1954460, 6.1611257
2: -7.5626879, 2.8519831, -3.3015814, 1.6984549, -9.2611427, 6.1535645
3: -6.6175451, 3.1929245, -2.9139428, 1.6769474, -8.2944927, 6.1068673
4: -6.8958654, 4.2538137, -3.2160358, 2.1296773, -9.0255432, 7.4698496
5: -5.2023110, 4.5596952, -2.4689925, 2.2197416, -7.4220524, 7.0286875
6: -5.4277568, 4.5908957, -2.5672934, 2.2093468, -7.6371036, 7.1581888
7: -6.2923150, 4.5697994, -2.9334710, 2.2764809, -8.5687962, 7.5032701
8: -7.1313362, 4.2114410, -3.3410714, 2.1363740, -9.2677097, 7.5525122
9: -5.0440254, 5.7778521, -2.4748447, 2.7623358, -7.8063612, 8.2526970

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324197
time: 3.58 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324192
time: 2.72 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2.7200260, 2.1661086, -2.8512540, 2.2627337, -4.9827595, 5.0173626
1: -2.1433887, 2.0247507, -2.2328126, 2.1082034, -4.2515922, 4.2575636
2: -3.3543868, 1.7220030, -3.5412455, 1.7594935, -5.1138802, 5.2632484
3: -2.9604073, 1.6977860, -3.1148162, 1.7615002, -4.7219076, 4.8126020
4: -3.2632613, 2.1607199, -3.4245656, 2.2476377, -5.5108991, 5.5852852
5: -2.5136003, 2.2481799, -2.6210511, 2.3473935, -4.8609939, 4.8692312
6: -2.6038549, 2.2434061, -2.7279384, 2.3417883, -4.9456434, 4.9713445
7: -2.9809713, 2.3109436, -3.1219184, 2.4031744, -5.3841457, 5.4328623
8: -3.3895664, 2.1683950, -3.5538847, 2.2512808, -5.6408472, 5.7222795
9: -2.5137594, 2.7942827, -2.6194108, 2.9310386, -5.4447980, 5.4136934

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
time: 2.69 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
time: 2.33 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -6.3205810, 4.8004994, -2.8037958, 2.2279718, -8.5485525, 7.6042953
1: -4.9374208, 4.3939800, -2.2000954, 2.0762708, -7.0136919, 6.5940752
2: -8.2714624, 3.2594941, -3.4745617, 1.7419813, -10.0134439, 6.7340555
3: -7.2546554, 3.4685085, -3.0567567, 1.7379328, -8.9925880, 6.5252652
4: -7.5177426, 4.6388021, -3.3670454, 2.2146854, -9.7324276, 8.0058479
5: -5.6542635, 4.9533744, -2.5787539, 2.3113594, -7.9656229, 7.5321283
6: -5.9270120, 5.0531297, -2.6836472, 2.3047366, -8.2317486, 7.7367768
7: -6.8490715, 5.0186992, -3.0696311, 2.3672628, -9.2163343, 8.0883303
8: -7.8645062, 4.5837641, -3.4948890, 2.2189007, -10.0834064, 8.0786533
9: -5.4922948, 6.3318939, -2.5792959, 2.8837731, -8.3760681, 8.9111900

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
time: 2.67 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
time: 2.13 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.9973176, 2.3697085, -2.8952935, 2.2936373, -5.2909546, 5.2650023
1: -2.3335009, 2.2069466, -2.2642684, 2.1427622, -4.4762630, 4.4712152
2: -3.7462654, 1.8130248, -3.6013370, 1.7856565, -5.5319219, 5.4143620
3: -3.2937617, 1.8342454, -3.1685982, 1.7848133, -5.0785751, 5.0028439
4: -3.6018212, 2.3500030, -3.4752483, 2.2829573, -5.8847785, 5.8252516
5: -2.7513871, 2.4577379, -2.6703582, 2.3790715, -5.1304588, 5.1280961
6: -2.8649039, 2.4561791, -2.7688076, 2.3800302, -5.2449341, 5.2249870
7: -3.2827492, 2.5135856, -3.1743832, 2.4417624, -5.7245116, 5.6879687
8: -3.7358642, 2.3512032, -3.6069160, 2.2877617, -6.0236259, 5.9581194
9: -2.7425733, 3.0774326, -2.6614912, 2.9672120, -5.7097855, 5.7389240

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324383, upper bound: 7.6327710
time: 3.48 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
time: 3.20 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.1214218, 2.4605050, -3.3594167, 2.6305923, -5.7520142, 5.8199215
1: -2.4187989, 2.2900453, -2.5864820, 2.4517391, -4.8705378, 4.8765273
2: -3.9193032, 1.8571093, -4.2425985, 1.9599543, -5.8792572, 6.0997076
3: -3.4465160, 1.8956268, -3.7323713, 2.0136664, -5.4601822, 5.6279984
4: -3.7534587, 2.4359410, -4.0356216, 2.6024401, -6.3558989, 6.4715624
5: -2.8613255, 2.5510793, -3.0788145, 2.7263789, -5.5877047, 5.6298938
6: -2.9815106, 2.5530663, -3.2031105, 2.7394447, -5.7209554, 5.7561769
7: -3.4191084, 2.6067133, -3.6790948, 2.7887216, -6.2078300, 6.2858081
8: -3.8909020, 2.4353621, -4.1837339, 2.5995026, -6.4904046, 6.6190958
9: -2.8471899, 3.2023582, -3.0494063, 3.4293561, -6.2765460, 6.2517643

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324556, upper bound: 7.6328217
time: 15.17 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324394, upper bound: 7.6324286
time: 2.80 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.9076688, 3.0339515, -2.8952935, 2.2936373, -6.2013063, 5.9292450
1: -3.0225821, 2.8135800, -2.2642684, 2.1427622, -5.1653442, 5.0778484
2: -4.9962831, 2.2021904, -3.6013370, 1.7856565, -6.7819395, 5.8035274
3: -4.3835006, 2.2860026, -3.1685982, 1.7848133, -6.1683140, 5.4546008
4: -4.6791110, 2.9837708, -3.4752483, 2.2829573, -6.9620686, 6.4590192
5: -3.5542026, 3.1377485, -2.6703582, 2.3790715, -5.9332743, 5.8081064
6: -3.7059753, 3.1675823, -2.7688076, 2.3800302, -6.0860052, 5.9363899
7: -4.2665973, 3.2041330, -3.1743832, 2.4417624, -6.7083597, 6.3785162
8: -4.8673449, 2.9676681, -3.6069160, 2.2877617, -7.1551065, 6.5745840
9: -3.5012918, 3.9700670, -2.6614912, 2.9672120, -6.4685040, 6.6315584

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B1_B1

### Relational analysis result of NS_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324311, upper bound: 7.6327734
time: 3.12 seconds

## Relational analysis of NS_B2_A2_B1_B2

### Relational analysis result of NS_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324095, upper bound: 7.6323609
time: 2.79 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -4.0467625, 3.1360638, -3.3594167, 2.6305923, -6.6773548, 6.4954805
1: -3.1339869, 2.9049907, -2.5864820, 2.4517391, -5.5857258, 5.4914727
2: -5.1853676, 2.2577319, -4.2425985, 1.9599543, -7.1453218, 6.5003304
3: -4.5500417, 2.3552206, -3.7323713, 2.0136664, -6.5637083, 6.0875921
4: -4.8448181, 3.0797195, -4.0356216, 2.6024401, -7.4472580, 7.1153412
5: -3.6740952, 3.2401381, -3.0788145, 2.7263789, -6.4004741, 6.3189526
6: -3.8351860, 3.2770946, -3.2031105, 2.7394447, -6.5746307, 6.4802051
7: -4.4155798, 3.3093321, -3.6790948, 2.7887216, -7.2043014, 6.9884272
8: -5.0399632, 3.0617819, -4.1837339, 2.5995026, -7.6394658, 7.2455158
9: -3.6164696, 4.1096478, -3.0494063, 3.4293561, -7.0458260, 7.1590538

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_B1

### Relational analysis result of NS_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324473, upper bound: 7.6328235
time: 2.65 seconds

## Relational analysis of NS_B2_A2_B2_B2

### Relational analysis result of NS_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324284, upper bound: 7.6324278
time: 2.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.23 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6323605, upper bound: 7.6324198
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324197
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324192
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324280, upper bound: 7.6324401
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
NS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324383, upper bound: 7.6327710
NS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324556, upper bound: 7.6328217
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324394, upper bound: 7.6324286
NS_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324311, upper bound: 7.6327734
NS_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324095, upper bound: 7.6323609
NS_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324473, upper bound: 7.6328235
NS_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.23
Output dim: 2, lower bound: -7.6324284, upper bound: 7.6324278

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -2.3151975, 1.8625641, -2.2380745, 1.8050854, -4.1202831, 4.1006384
1: -1.8558949, 1.7448099, -1.7980882, 1.6848613, -3.5407562, 3.5428982
2: -2.7631500, 1.5726111, -2.6492724, 1.5403659, -4.3035159, 4.2218838
3: -2.4932914, 1.4910572, -2.3990428, 1.4490991, -3.9423904, 3.8901000
4: -2.7511826, 1.8712901, -2.6521759, 1.8108498, -4.5620322, 4.5234661
5: -2.1427438, 1.9427977, -2.0623641, 1.8858886, -4.0286322, 4.0051618
6: -2.2092307, 1.9236181, -2.1315165, 1.8590215, -4.0682521, 4.0551348
7: -2.5191453, 2.0075314, -2.4316661, 1.9428017, -4.4619470, 4.4391975
8: -2.8667698, 1.8952205, -2.7659321, 1.8354522, -4.7022219, 4.6611528
9: -2.1585655, 2.3879640, -2.0822058, 2.3160677, -4.4746332, 4.4701700

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281447
time: 4.67 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320199
time: 5.68 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -2.3151975, 1.8625641, -5.6185951, 4.2716193, -6.5868168, 7.4811592
1: -1.8558949, 1.7448099, -4.1213198, 3.9566376, -5.8125324, 5.8661299
2: -2.7631500, 1.5726111, -7.4328771, 2.7999535, -5.5631037, 9.0054884
3: -2.4932914, 1.4910572, -6.1725292, 3.1276188, -5.6209102, 7.6635866
4: -2.7511826, 1.8712901, -6.7716045, 4.1562223, -6.9074049, 8.6428947
5: -2.1427438, 1.9427977, -5.0997190, 4.4419203, -6.5846643, 7.0425167
6: -2.2092307, 1.9236181, -5.3159370, 4.4713960, -6.6806269, 7.2395554
7: -2.5191453, 2.0075314, -6.1811018, 4.4241567, -6.9433022, 8.1886330
8: -2.8667698, 1.8952205, -6.9815183, 4.0893936, -6.9561634, 8.8767385
9: -2.1585655, 2.3879640, -4.9549112, 5.5824480, -7.7410135, 7.3428755

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281441
time: 6.54 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320198
time: 5.61 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -5.7190075, 4.3689775, -2.2380745, 1.8050854, -7.5240927, 6.6070518
1: -4.2014623, 4.0458031, -1.7980882, 1.6848613, -5.8863235, 5.8438911
2: -7.5626879, 2.8519831, -2.6492724, 1.5403659, -9.1030540, 5.5012555
3: -6.6175451, 3.1929245, -2.3990428, 1.4490991, -8.0666447, 5.5919676
4: -6.8958654, 4.2538137, -2.6521759, 1.8108498, -8.7067156, 6.9059896
5: -5.2023110, 4.5596952, -2.0623641, 1.8858886, -7.0881996, 6.6220593
6: -5.4277568, 4.5908957, -2.1315165, 1.8590215, -7.2867785, 6.7224121
7: -6.2923150, 4.5697994, -2.4316661, 1.9428017, -8.2351170, 7.0014658
8: -7.1313362, 4.2114410, -2.7659321, 1.8354522, -8.9667883, 6.9773731
9: -5.0440254, 5.7778521, -2.0822058, 2.3160677, -7.3600931, 7.8600578

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283727, upper bound: 7.6304619
time: 3.07 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319270, upper bound: 7.6320006
time: 3.76 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -5.7190075, 4.3689775, -5.6185951, 4.2716193, -9.9906273, 9.9875727
1: -4.2014623, 4.0458031, -4.1213198, 3.9566376, -8.1581001, 8.1671228
2: -7.5626879, 2.8519831, -7.4328771, 2.7999535, -10.3626413, 10.2848606
3: -6.6175451, 3.1929245, -6.1725292, 3.1276188, -9.7451639, 9.3654537
4: -6.8958654, 4.2538137, -6.7716045, 4.1562223, -11.0520878, 11.0254183
5: -5.2023110, 4.5596952, -5.0997190, 4.4419203, -9.6442318, 9.6594143
6: -5.4277568, 4.5908957, -5.3159370, 4.4713960, -9.8991528, 9.9068327
7: -6.2923150, 4.5697994, -6.1811018, 4.4241567, -10.7164717, 10.7509012
8: -7.1313362, 4.2114410, -6.9815183, 4.0893936, -11.2207298, 11.1929588
9: -5.0440254, 5.7778521, -4.9549112, 5.5824480, -10.6264734, 10.7327633

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324198
time: 2.61 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324191
time: 3.27 seconds

## BFS NS instance: NS_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2.7200260, 2.1661086, -2.3338535, 1.8806324, -4.6006584, 4.4999619
1: -2.1433887, 2.0247507, -1.8705406, 1.7552400, -3.8986287, 3.8952913
2: -3.3543868, 1.7220030, -2.7982168, 1.5723163, -4.9267030, 4.5202198
3: -2.9604073, 1.6977860, -2.5174484, 1.5006043, -4.4610114, 4.2152343
4: -3.2632613, 2.1607199, -2.7822888, 1.8823733, -5.1456347, 4.9430084
5: -2.5136003, 2.2481799, -2.1522994, 1.9604743, -4.4740744, 4.4004793
6: -2.6038549, 2.2434061, -2.2313132, 1.9377041, -4.5415592, 4.4747190
7: -2.9809713, 2.3109436, -2.5428867, 2.0178661, -4.9988375, 4.8538303
8: -3.3895664, 2.1683950, -2.8974190, 1.9023521, -5.2919188, 5.0658140
9: -2.5137594, 2.7942827, -2.1729269, 2.4178085, -4.9315681, 4.9672098

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A1_B1_B1

### Relational analysis result of NS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281819
time: 2.80 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2

### Relational analysis result of NS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320435
time: 4.79 seconds

## BFS NS instance: NS_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.7200260, 2.1661086, -5.7596278, 4.3945508, -7.1145768, 7.9257364
1: -2.1433887, 2.0247507, -4.2299485, 4.0584455, -6.2018342, 6.2546992
2: -3.3543868, 1.7220030, -7.6227016, 2.8566599, -6.2110467, 9.3447046
3: -2.9604073, 1.6977860, -6.6704755, 3.2000456, -6.1604528, 8.3682613
4: -3.2632613, 2.1607199, -6.9428921, 4.2715836, -7.5348449, 9.1036119
5: -2.5136003, 2.2481799, -5.2225118, 4.5900369, -7.1036372, 7.4706917
6: -2.6038549, 2.2434061, -5.4462862, 4.6206875, -7.2245426, 7.6896925
7: -2.9809713, 2.3109436, -6.3334379, 4.5902991, -7.5712705, 8.6443815
8: -3.3895664, 2.1683950, -7.1581602, 4.2342615, -7.6238279, 9.3265553
9: -2.5137594, 2.7942827, -5.0722699, 5.8229094, -8.3366690, 7.8665524

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A1_B2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281815
time: 3.03 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320434
time: 4.95 seconds

## BFS NS instance: NS_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -6.3205810, 4.8004994, -2.3338535, 1.8806324, -8.2012138, 7.1343527
1: -4.9374208, 4.3939800, -1.8705406, 1.7552400, -6.6926608, 6.2645206
2: -8.2714624, 3.2594941, -2.7982168, 1.5723163, -9.8437786, 6.0577106
3: -7.2546554, 3.4685085, -2.5174484, 1.5006043, -8.7552595, 5.9859571
4: -7.5177426, 4.6388021, -2.7822888, 1.8823733, -9.4001160, 7.4210911
5: -5.6542635, 4.9533744, -2.1522994, 1.9604743, -7.6147375, 7.1056738
6: -5.9270120, 5.0531297, -2.2313132, 1.9377041, -7.8647161, 7.2844429
7: -6.8490715, 5.0186992, -2.5428867, 2.0178661, -8.8669376, 7.5615859
8: -7.8645062, 4.5837641, -2.8974190, 1.9023521, -9.7668581, 7.4811831
9: -5.4922948, 6.3318939, -2.1729269, 2.4178085, -7.9101033, 8.5048208

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A2_B1_A1

### Relational analysis result of NS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261535, upper bound: 7.6304678
time: 4.83 seconds

## Relational analysis of NS_B1_A2_A2_B1_A2

### Relational analysis result of NS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320165, upper bound: 7.6320278
time: 3.47 seconds

## BFS NS instance: NS_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -6.3205810, 4.8004994, -5.7575884, 4.3945508, -10.7151318, 10.5580883
1: -4.9374208, 4.3939800, -4.2299485, 4.0489388, -8.9863596, 8.6239281
2: -8.2714624, 3.2594941, -7.6227016, 2.8517237, -11.1231861, 10.8821955
3: -7.2546554, 3.4685085, -6.6704755, 3.1960638, -10.4507189, 10.1389837
4: -7.5177426, 4.6388021, -6.9428921, 4.2515326, -11.7692757, 11.5816936
5: -5.6542635, 4.9533744, -5.2222462, 4.5900369, -10.2443008, 10.1756210
6: -5.9270120, 5.0531297, -5.4462862, 4.6091986, -10.5362110, 10.4994164
7: -6.8490715, 5.0186992, -6.3334379, 4.5891066, -11.4381781, 11.3521366
8: -7.8645062, 4.5837641, -7.1542482, 4.2342615, -12.0987682, 11.7380123
9: -5.4922948, 6.3318939, -5.0722699, 5.8165703, -11.3088646, 11.4041634

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B1_A2_A2_B2_A1

### Relational analysis result of NS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
time: 3.42 seconds

## Relational analysis of NS_B1_A2_A2_B2_A2

### Relational analysis result of NS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324396
time: 1.94 seconds

## BFS NS instance: NS_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2.7276950, 2.1729178, -2.2355657, 1.7990253, -4.5267200, 4.4084835
1: -2.1480026, 2.0260634, -1.7956465, 1.6860437, -3.8340464, 3.8217101
2: -3.3686826, 1.7159586, -2.6386285, 1.5432963, -4.9119787, 4.3545871
3: -2.9669166, 1.7006347, -2.3953915, 1.4480960, -4.4150124, 4.0960264
4: -3.2738788, 2.1627779, -2.6439750, 1.8106759, -5.0845547, 4.8067532
5: -2.5115216, 2.2552068, -2.0660846, 1.8807189, -4.3922405, 4.3212914
6: -2.6118550, 2.2460961, -2.1259298, 1.8584037, -4.4702587, 4.3720260
7: -2.9860487, 2.3115568, -2.4252632, 1.9449842, -4.9310331, 4.7368202
8: -3.4003899, 2.1680882, -2.7569976, 1.8393042, -5.2396941, 4.9250860
9: -2.5151827, 2.8083096, -2.0834792, 2.3034928, -4.8186755, 4.8917885

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A1_B1_B1_A1

### Relational analysis result of NS_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323606
time: 11.17 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2

### Relational analysis result of NS_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323611
time: 4.10 seconds

## BFS NS instance: NS_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2.6799695, 2.1383300, -5.6204772, 4.2910242, -6.9709940, 7.7588072
1: -2.1153226, 1.9939835, -4.1353722, 3.9699244, -6.0852470, 6.1293554
2: -3.3015814, 1.6984549, -7.4287090, 2.8142042, -6.1157856, 9.1271639
3: -2.9139428, 1.6769474, -6.4994969, 3.1324167, -6.0463595, 8.1764441
4: -3.2160358, 2.1296773, -6.7710991, 4.1788130, -7.3948488, 8.9007759
5: -2.4689925, 2.2197416, -5.1077037, 4.4831533, -6.9521456, 7.3274450
6: -2.5672934, 2.2093468, -5.3151417, 4.5160017, -7.0832949, 7.5244884
7: -2.9334710, 2.2764809, -6.1839952, 4.4906549, -7.4241257, 8.4604759
8: -3.3410714, 2.1363740, -6.9841442, 4.1458650, -7.4869366, 9.1205177
9: -2.4748447, 2.7623358, -4.9601774, 5.6757460, -8.1505909, 7.7225132

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
time: 2.67 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323605
time: 4.01 seconds

## BFS NS instance: NS_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -2.8512540, 2.2627337, -2.6196442, 2.0933833, -4.9446373, 4.8823776
1: -2.2328126, 2.1082034, -2.0746675, 1.9575135, -4.1903262, 4.1828709
2: -3.5412455, 1.7594935, -3.2131815, 1.6855036, -5.2267489, 4.9726748
3: -3.1148162, 1.7615002, -2.8486047, 1.6480401, -4.7628565, 4.6101050
4: -3.4245656, 2.2476377, -3.1411972, 2.0911605, -5.5157261, 5.3888350
5: -2.6210511, 2.3473935, -2.4237320, 2.1737525, -4.7948036, 4.7711258
6: -2.7279384, 2.3417883, -2.5094218, 2.1663046, -4.8942432, 4.8512101
7: -3.1219184, 2.4031744, -2.8701410, 2.2375689, -5.3594875, 5.2733154
8: -3.5538847, 2.2512808, -3.2649107, 2.1014581, -5.6553431, 5.5161915
9: -2.6194108, 2.9310386, -2.4290111, 2.6969142, -5.3163252, 5.3600497

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324286
time: 2.60 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324287
time: 12.61 seconds

## BFS NS instance: NS_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2.8037958, 2.2279718, -6.1708174, 4.6784267, -7.4822226, 8.3987894
1: -2.2000954, 2.0762708, -4.8206434, 4.2905436, -6.4906387, 6.8969145
2: -3.4745617, 1.7419813, -8.0471668, 3.1853733, -6.6599350, 9.7891483
3: -3.0567567, 1.7379328, -7.0704432, 3.3784189, -6.4351759, 8.8083763
4: -3.3670454, 2.2146854, -7.3469925, 4.5044041, -7.8714495, 9.5616779
5: -2.5787539, 2.3113594, -5.5247989, 4.8391924, -7.4179463, 7.8361583
6: -2.6836472, 2.3047366, -5.7866602, 4.9247112, -7.6083584, 8.0913963
7: -3.0696311, 2.3672628, -6.6892972, 4.9057126, -7.9753437, 9.0565605
8: -3.4948890, 2.2189007, -7.6365604, 4.4801340, -7.9750233, 9.8554611
9: -2.5792959, 2.8837731, -5.3680449, 6.1882858, -8.7675819, 8.2518177

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A1_B2_B2_A1

### Relational analysis result of NS_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324281
time: 3.36 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2

### Relational analysis result of NS_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324286
time: 2.81 seconds

## BFS NS instance: NS_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.6245940, 2.8274207, -2.2355657, 1.7990253, -5.4236193, 5.0629864
1: -2.7959232, 2.6270945, -1.7956465, 1.6860437, -4.4819670, 4.4227409
2: -4.6116285, 2.0867095, -2.6386285, 1.5432963, -6.1549249, 4.7253380
3: -4.0458851, 2.1457882, -2.3953915, 1.4480960, -5.4939814, 4.5411797
4: -4.3439732, 2.7887988, -2.6439750, 1.8106759, -6.1546488, 5.4327736
5: -3.3095040, 2.9292896, -2.0660846, 1.8807189, -5.1902227, 4.9953742
6: -3.4430983, 2.9461675, -2.1259298, 1.8584037, -5.3015022, 5.0720973
7: -3.9634364, 2.9904122, -2.4252632, 1.9449842, -5.9084206, 5.4156752
8: -4.5167484, 2.7771845, -2.7569976, 1.8393042, -6.3560524, 5.5341821
9: -3.2673078, 3.6875274, -2.0834792, 2.3034928, -5.5708008, 5.7710066

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B1_B1_A1

### Relational analysis result of NS_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
time: 2.97 seconds

## Relational analysis of NS_B2_A2_B1_B1_A2

### Relational analysis result of NS_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
time: 4.34 seconds

## BFS NS instance: NS_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.5749564, 2.7911797, -5.6204772, 4.2910242, -7.8659806, 8.4116573
1: -2.7562363, 2.5942686, -4.1353722, 3.9699244, -6.7261610, 6.7296410
2: -4.5440865, 2.0658956, -7.4287090, 2.8142042, -7.3582907, 9.4946041
3: -3.9869032, 2.1212063, -6.4994969, 3.1324167, -7.1193199, 8.6207027
4: -4.2853813, 2.7546802, -6.7710991, 4.1788130, -8.4641943, 9.5257797
5: -3.2665367, 2.8925681, -5.1077037, 4.4831533, -7.7496901, 8.0002718
6: -3.3974385, 2.9073250, -5.3151417, 4.5160017, -7.9134402, 8.2224665
7: -3.9103365, 2.9527512, -6.1839952, 4.4906549, -8.4009914, 9.1367464
8: -4.4551463, 2.7438636, -6.9841442, 4.1458650, -8.6010113, 9.7280083
9: -3.2262805, 3.6382346, -4.9601774, 5.6757460, -8.9020262, 8.5984116

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B1_B2_A1

### Relational analysis result of NS_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
time: 3.31 seconds

## Relational analysis of NS_B2_A2_B1_B2_A2

### Relational analysis result of NS_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
time: 4.33 seconds

## BFS NS instance: NS_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3.7643478, 2.9293232, -2.6196442, 2.0933833, -5.8577309, 5.5489674
1: -2.9076014, 2.7188129, -2.0746675, 1.9575135, -4.8651147, 4.7934804
2: -4.8008823, 2.1425965, -3.2131815, 1.6855036, -6.4863858, 5.3557777
3: -4.2132230, 2.2147810, -2.8486047, 1.6480401, -5.8612633, 5.0633860
4: -4.5102758, 2.8844206, -3.1411972, 2.0911605, -6.6014366, 6.0256176
5: -3.4300013, 3.0320659, -2.4237320, 2.1737525, -5.6037540, 5.4557981
6: -3.5729051, 3.0552044, -2.5094218, 2.1663046, -5.7392097, 5.5646262
7: -4.1131449, 3.0956268, -2.8701410, 2.2375689, -6.3507137, 5.9657679
8: -4.6902037, 2.8706493, -3.2649107, 2.1014581, -6.7916617, 6.1355600
9: -3.3830318, 3.8275721, -2.4290111, 2.6969142, -6.0799460, 6.2565832

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_B1_A1

### Relational analysis result of NS_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324284
time: 4.11 seconds

## Relational analysis of NS_B2_A2_B2_B1_A2

### Relational analysis result of NS_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324278
time: 7.16 seconds

## BFS NS instance: NS_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.7136314, 2.8923538, -6.1708174, 4.6784267, -8.3920584, 9.0631714
1: -2.8671050, 2.6852334, -4.8206434, 4.2905436, -7.1576486, 7.5058765
2: -4.7320290, 2.1213841, -8.0471668, 3.1853733, -7.9174023, 10.1685505
3: -4.1530290, 2.1896510, -7.0704432, 3.3784189, -7.5314479, 9.2600937
4: -4.4504952, 2.8494086, -7.3469925, 4.5044041, -8.9548988, 10.1964016
5: -3.3861663, 2.9945903, -5.5247989, 4.8391924, -8.2253590, 8.5193892
6: -3.5260615, 3.0155828, -5.7866602, 4.9247112, -8.4507732, 8.8022432
7: -4.0588980, 3.0571866, -6.6892972, 4.9057126, -8.9646111, 9.7464838
8: -4.6273704, 2.8366013, -7.6365604, 4.4801340, -9.1075039, 10.4731617
9: -3.3410745, 3.7772696, -5.3680449, 6.1882858, -9.5293598, 9.1453142

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of NS_B2_A2_B2_B2_A1

### Relational analysis result of NS_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324277
time: 3.07 seconds

## Relational analysis of NS_B2_A2_B2_B2_A2

### Relational analysis result of NS_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324284
time: 2.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.76 seconds
NS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281447
NS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320199
NS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6307164, upper bound: 7.6281441
NS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6323442, upper bound: 7.6320198
NS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6283727, upper bound: 7.6304619
NS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6319270, upper bound: 7.6320006
NS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324198
NS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6323611, upper bound: 7.6324191
NS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281819
NS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320435
NS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6307375, upper bound: 7.6281815
NS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324225, upper bound: 7.6320434
NS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6261535, upper bound: 7.6304678
NS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6320165, upper bound: 7.6320278
NS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324401
NS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324286, upper bound: 7.6324396
NS_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323606
NS_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324197, upper bound: 7.6323611
NS_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323612
NS_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324191, upper bound: 7.6323605
NS_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324286
NS_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324401, upper bound: 7.6324287
NS_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324281
NS_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324395, upper bound: 7.6324286
NS_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
NS_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324089, upper bound: 7.6323609
NS_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
NS_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6323609
NS_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324284
NS_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324279, upper bound: 7.6324278
NS_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324277
NS_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.76
Output dim: 2, lower bound: -7.6324278, upper bound: 7.6324284

## BFS NS instance: NS_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.5742193, 0.6161574, -0.1558935, 0.1683881, -0.7426074, 0.7720509
1: -0.5799464, 0.5992749, -0.2020659, 0.2123953, -0.7923417, 0.8013408
2: -0.0456628, 1.1171672, 0.6011345, 1.0350590, -1.0807219, 0.5160328
3: -0.4268622, 0.6182083, -0.0762342, 0.2830463, -0.7099085, 0.6944425
4: -0.6577558, 0.6442816, -0.2180401, 0.2160968, -0.8738526, 0.8623216
5: -0.5705507, 0.6804613, -0.1963118, 0.2151469, -0.7856975, 0.8767731
6: -0.5334853, 0.6385880, -0.1595097, 0.2242475, -0.7577328, 0.7980976
7: -0.6026897, 0.7060025, -0.1881549, 0.2411661, -0.8438557, 0.8941574
8: -0.6733687, 0.7379457, -0.2117272, 0.3074552, -0.9808239, 0.9496729
9: -0.6323479, 0.6944107, -0.2238538, 0.1974250, -0.8297729, 0.9182645

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6300020, upper bound: 7.5860490
time: 3.63 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302489, upper bound: 7.5860504
time: 2.48 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2.0870566, 1.6858394, -1.3644453, 1.1501882, -3.2372448, 3.0502849
1: -1.6829654, 1.5765797, -1.1318805, 1.0669785, -2.7499437, 2.7084603
2: -2.3958759, 1.4946616, -1.2343626, 1.2828192, -3.6786952, 2.7290242
3: -2.2124774, 1.3681638, -1.3378156, 0.9844181, -3.1968956, 2.7059793
4: -2.4623585, 1.6999377, -1.5969075, 1.1730683, -3.6354268, 3.2968452
5: -1.9270306, 1.7584631, -1.2479331, 1.2082707, -3.1353011, 3.0063963
6: -1.9723170, 1.7388371, -1.2479024, 1.1776177, -3.1499348, 2.9867396
7: -2.2560096, 1.8258598, -1.4493136, 1.2553008, -3.5113103, 3.2751734
8: -2.5543363, 1.7349540, -1.5698447, 1.2372620, -3.7915983, 3.3047986
9: -1.9438657, 2.1525607, -1.2857995, 1.4212538, -3.3651195, 3.4383602

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B1_B2_A1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305727, upper bound: 7.6315649
time: 2.83 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305728, upper bound: 7.6326769
time: 3.60 seconds

## BFS NS instance: NS_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.5742193, 0.6161574, -1.4339643, 1.1984482, -1.7726674, 2.0501218
1: -0.5799464, 0.5992749, -1.1851937, 1.1114624, -1.6914088, 1.7844685
2: -0.0456628, 1.1171672, -1.3429151, 1.2996322, -1.3452950, 2.4600823
3: -0.4268622, 0.6182083, -1.4198550, 1.0189645, -1.4458268, 2.0380633
4: -0.6577558, 0.6442816, -1.6753489, 1.2212707, -1.8790264, 2.3196304
5: -0.5705507, 0.6804613, -1.3111669, 1.2524818, -1.8230325, 1.9916282
6: -0.5334853, 0.6385880, -1.3173723, 1.2309831, -1.7644684, 1.9559603
7: -0.6026897, 0.7060025, -1.5275031, 1.3041041, -1.9067938, 2.2335057
8: -0.6733687, 0.7379457, -1.6616356, 1.2835367, -1.9569054, 2.3995814
9: -0.6323479, 0.6944107, -1.3437171, 1.4875968, -2.1199446, 2.0381279

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B2_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290859, upper bound: 7.4743615
time: 2.92 seconds

## Relational analysis of NS_B1_A1_A1_B2_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289733, upper bound: 7.4419433
time: 2.96 seconds

## BFS NS instance: NS_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2.0870566, 1.6858394, -4.5606079, 3.5044854, -5.5915422, 6.2464476
1: -1.6829654, 1.5765797, -3.3987250, 3.2476947, -4.9306602, 4.9753046
2: -2.3958759, 1.4946616, -5.9579086, 2.4050877, -4.8009634, 7.4525700
3: -2.2124774, 1.3681638, -5.0008011, 2.6033912, -4.8158684, 6.3689651
4: -2.4623585, 1.6999377, -5.4930792, 3.4236238, -5.8859825, 7.1930170
5: -1.9270306, 1.7584631, -4.1529512, 3.6440611, -5.5710917, 5.9114141
6: -1.9723170, 1.7388371, -4.3277369, 3.6579456, -5.6302624, 6.0665741
7: -2.2560096, 1.8258598, -5.0142360, 3.6484683, -5.9044781, 6.8400955
8: -2.5543363, 1.7349540, -5.6715260, 3.3842456, -5.9385819, 7.4064798
9: -1.9438657, 2.1525607, -4.0622034, 4.5646563, -6.5085220, 6.2147641

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316419, upper bound: 7.6309815
time: 4.64 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317932, upper bound: 7.6314472
time: 2.86 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.5729212, 1.3003097, -0.5290306, 0.5901527, -2.1630738, 1.8293402
1: -1.2916257, 1.2064134, -0.5534776, 0.5771359, -1.8687615, 1.7598910
2: -1.5531341, 1.3401685, 0.0126180, 1.1089032, -2.6620374, 1.3275505
3: -1.5894464, 1.0895388, -0.3844105, 0.6012738, -2.1907203, 1.4739493
4: -1.8438501, 1.3189608, -0.6121894, 0.6145369, -2.4583871, 1.9311502
5: -1.4452966, 1.3476498, -0.5431017, 0.6472400, -2.0925367, 1.8907515
6: -1.4485787, 1.3399221, -0.4972519, 0.6147041, -2.0632830, 1.8371739
7: -1.6846026, 1.4115483, -0.5641073, 0.6789837, -2.3635864, 1.9756556
8: -1.8512034, 1.3800359, -0.6342140, 0.7126825, -2.5638859, 2.0142498
9: -1.4756049, 1.6203052, -0.6002016, 0.6578673, -2.1334722, 2.2205067

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A2_B1_A1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4878332, upper bound: 7.6293744
time: 2.07 seconds

## Relational analysis of NS_B1_A1_A2_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4224551, upper bound: 7.6292976
time: 3.45 seconds

## BFS NS instance: NS_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -4.6485190, 3.5828099, -2.0160203, 1.6369640, -6.2854829, 5.5988302
1: -3.4665895, 3.3214092, -1.6295309, 1.5221844, -4.9887738, 4.9509401
2: -6.0722737, 2.4485002, -2.2899611, 1.4721330, -7.5444069, 4.7384615
3: -5.3124390, 2.6569118, -2.1254168, 1.3296511, -6.6420898, 4.7823286
4: -5.6009140, 3.5029910, -2.3793466, 1.6467444, -7.2476583, 5.8823376
5: -4.2426496, 3.7354851, -1.8554683, 1.7073951, -5.9500446, 5.5909534
6: -4.4214711, 3.7535670, -1.9012587, 1.6823790, -6.1038504, 5.6548257
7: -5.1125979, 3.7600589, -2.1823444, 1.7655002, -6.8780980, 5.9424033
8: -5.7955399, 3.4800866, -2.4629154, 1.6820664, -7.4776063, 5.9430017
9: -4.1408706, 4.7101393, -1.8773210, 2.0877132, -6.2285838, 6.5874605

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6299327
time: 5.28 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6324261
time: 12.49 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -4.7120776, 3.6348763, -5.6185951, 4.2716193, -8.9836969, 9.2534714
1: -3.5086536, 3.3622956, -4.1213198, 3.9566376, -7.4652910, 7.4836154
2: -6.1824889, 2.4841852, -7.4328771, 2.7999535, -8.9824429, 9.9170628
3: -5.3841496, 2.6883509, -6.1725292, 3.1276188, -8.5117683, 8.8608799
4: -5.6728554, 3.5463605, -6.7716045, 4.1562223, -9.8290777, 10.3179646
5: -4.2994499, 3.8076627, -5.0997190, 4.4419203, -8.7413702, 8.9073820
6: -4.4795990, 3.8009748, -5.3159370, 4.4713960, -8.9509945, 9.1169119
7: -5.1841855, 3.8064499, -6.1811018, 4.4241567, -9.6083422, 9.9875517
8: -5.8723760, 3.5217280, -6.9815183, 4.0893936, -9.9617691, 10.5032463
9: -4.1889925, 4.7741551, -4.9549112, 5.5824480, -9.7714405, 9.7290668

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A2_B2_A1_A1

### Relational analysis result of NS_B1_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302952, upper bound: 7.6290127
time: 3.39 seconds

## Relational analysis of NS_B1_A1_A2_B2_A1_A2

### Relational analysis result of NS_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6288282, upper bound: 7.6289555
time: 4.29 seconds

## BFS NS instance: NS_B1_A1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.6229591, 4.2989349, -5.6185951, 4.2716193, -9.8945789, 9.9175301
1: -4.1353722, 3.9811015, -4.1213198, 3.9566376, -8.0920095, 8.1024208
2: -7.4313059, 2.8179891, -7.4328771, 2.7999535, -10.2312593, 10.2508659
3: -6.4994969, 3.1449418, -6.1725292, 3.1276188, -9.6271152, 9.3174706
4: -6.7787857, 4.1866455, -6.7716045, 4.1562223, -10.9350080, 10.9582500
5: -5.1166859, 4.4884729, -5.0997190, 4.4419203, -9.5586061, 9.5881920
6: -5.3368411, 4.5160017, -5.3159370, 4.4713960, -9.8082371, 9.8319387
7: -6.1868453, 4.4975691, -6.1811018, 4.4241567, -10.6110020, 10.6786709
8: -7.0113182, 4.1458650, -6.9815183, 4.0893936, -11.1007118, 11.1273832
9: -4.9629936, 5.6809969, -4.9549112, 5.5824480, -10.5454416, 10.6359081

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_B1_A1_A2_B2_A2_A1

### Relational analysis result of NS_B1_A1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323388, upper bound: 7.6323961
time: 3.47 seconds

## Relational analysis of NS_B1_A1_A2_B2_A2_A2

### Relational analysis result of NS_B1_A1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323366, upper bound: 7.6323964
time: 5.06 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.8380679, 0.7874784, -0.1623101, 0.1752629, -1.0133308, 0.9497885
1: -0.7495179, 0.7397859, -0.2093137, 0.2204718, -0.9699897, 0.9490996
2: -0.4249045, 1.1632850, 0.5897677, 1.0350578, -1.4599622, 0.5735173
3: -0.7033481, 0.7291363, -0.0815222, 0.2916855, -0.9950336, 0.8106585
4: -0.9612570, 0.8164982, -0.2258524, 0.2216960, -1.1829530, 1.0423506
5: -0.7759328, 0.8592340, -0.2038466, 0.2235304, -0.9994631, 1.0630807
6: -0.7561188, 0.8049411, -0.1647633, 0.2340025, -0.9901213, 0.9697043
7: -0.8623356, 0.8683124, -0.1936130, 0.2502241, -1.1125597, 1.0619254
8: -0.9263675, 0.8941428, -0.2202273, 0.3167633, -1.2431308, 1.1143701
9: -0.8332343, 0.9297156, -0.2322594, 0.2051974, -1.0384316, 1.1619750

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_A1_B1_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6300528, upper bound: 7.5861040
time: 3.63 seconds

## Relational analysis of NS_B1_A2_A1_B1_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303786, upper bound: 7.5861222
time: 3.64 seconds

## BFS NS instance: NS_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2.4637976, 1.9779339, -1.4592549, 1.2207263, -3.6845238, 3.4371886
1: -1.9675179, 1.8516169, -1.2046081, 1.1302004, -3.0977182, 3.0562248
2: -2.9840589, 1.6205074, -1.3846933, 1.3090674, -4.2931261, 3.0052006
3: -2.6755011, 1.5698578, -1.4540393, 1.0333321, -3.7088332, 3.0238972
4: -2.9522014, 1.9818386, -1.7119064, 1.2399557, -4.1921568, 3.6937451
5: -2.2806354, 2.0506394, -1.3359098, 1.2751590, -3.5557942, 3.3865492
6: -2.3630037, 2.0457332, -1.3397732, 1.2494270, -3.6124306, 3.3855064
7: -2.6945055, 2.1217723, -1.5558772, 1.3278602, -4.0223656, 3.6776495
8: -3.0702014, 1.9966767, -1.6996015, 1.3014858, -4.3716869, 3.6962781
9: -2.2962828, 2.5471468, -1.3710129, 1.5176690, -3.8139517, 3.9181597

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A1_B1_B2_A1

### Relational analysis result of NS_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6315650
time: 2.26 seconds

## Relational analysis of NS_B1_A2_A1_B1_B2_A2

### Relational analysis result of NS_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303594, upper bound: 7.6326976
time: 3.39 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.8380679, 0.7874784, -1.5092630, 1.2556083, -2.0936761, 2.2967415
1: -0.7495179, 0.7397859, -1.2431486, 1.1624177, -1.9119356, 1.9829345
2: -0.4249045, 1.1632850, -1.4624932, 1.3207645, -1.7456690, 2.6257782
3: -0.7033481, 0.7291363, -1.5121390, 1.0581765, -1.7615247, 2.2412753
4: -0.9612570, 0.8164982, -1.7669803, 1.2748854, -2.2361424, 2.5834785
5: -0.7759328, 0.8592340, -1.3823025, 1.3060142, -2.0819468, 2.2415366
6: -0.7561188, 0.8049411, -1.3903584, 1.2886757, -2.0447946, 2.1952996
7: -0.8623356, 0.8683124, -1.6135566, 1.3615885, -2.2239242, 2.4818690
8: -0.9263675, 0.8941428, -1.7648863, 1.3349650, -2.2613325, 2.6590290
9: -0.8332343, 0.9297156, -1.4125023, 1.5650142, -2.3982487, 2.3422179

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_A1_B2_B1_B1

### Relational analysis result of NS_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6293385, upper bound: 7.4749466
time: 3.24 seconds

## Relational analysis of NS_B1_A2_A1_B2_B1_B2

### Relational analysis result of NS_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292978, upper bound: 7.4420164
time: 6.78 seconds

## BFS NS instance: NS_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2.4637976, 1.9779339, -4.6796112, 3.6039925, -6.0677900, 6.6575451
1: -1.9675179, 1.8516169, -3.4879055, 3.3318214, -5.2993393, 5.3395224
2: -2.9840589, 1.6205074, -6.1202345, 2.4515746, -5.4356337, 7.7407417
3: -2.6755011, 1.5698578, -5.3519697, 2.6638720, -5.3393731, 6.9218273
4: -2.9522014, 1.9818386, -5.6376638, 3.5168540, -6.4690552, 7.6195025
5: -2.2806354, 2.0506394, -4.2570877, 3.7608597, -6.0414953, 6.3077269
6: -2.3630037, 2.0457332, -4.4381380, 3.7753828, -6.1383867, 6.4838715
7: -2.6945055, 2.1217723, -5.1433854, 3.7758074, -6.4703131, 7.2651577
8: -3.0702014, 1.9966767, -5.8202367, 3.4960437, -6.5662451, 7.8169136
9: -2.2962828, 2.5471468, -4.1615582, 4.7464414, -7.0427241, 6.7087049

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6317450, upper bound: 7.6309989
time: 2.63 seconds

## Relational analysis of NS_B1_A2_A1_B2_B2_B2

### Relational analysis result of NS_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319177, upper bound: 7.6314989
time: 2.52 seconds

## BFS NS instance: NS_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.8952280, 1.5408925, -0.5749281, 0.6165418, -2.5117698, 2.1158206
1: -1.5365030, 1.4320529, -0.5802902, 0.5992781, -2.1357810, 2.0123429
2: -2.0620646, 1.4333646, -0.0465086, 1.1170228, -3.1790874, 1.4798732
3: -1.9795961, 1.2601513, -0.4274951, 0.6183562, -2.5979524, 1.6876464
4: -2.2320299, 1.5530980, -0.6582962, 0.6446346, -2.8766646, 2.2113941
5: -1.7470505, 1.5851779, -0.5711213, 0.6807762, -2.4278266, 2.1562991
6: -1.7761716, 1.5907991, -0.5341682, 0.6387562, -2.4149277, 2.1249673
7: -2.0472651, 1.6634418, -0.6032372, 0.7061540, -2.7534189, 2.2666790
8: -2.2936382, 1.5987157, -0.6736836, 0.7382634, -3.0319016, 2.2723992
9: -1.7685250, 1.9517260, -0.6324757, 0.6950939, -2.4636188, 2.5842018

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_B1_A2_A2_B1_A1_A1

### Relational analysis result of NS_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4716106, upper bound: 7.6294428
time: 2.81 seconds

## Relational analysis of NS_B1_A2_A2_B1_A1_A2

### Relational analysis result of NS_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.4404536, upper bound: 7.6294079
time: 2.02 seconds

## BFS NS instance: NS_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.1909037, 3.9723678, -2.1138093, 1.7098024, -6.9007063, 6.0861769
1: -4.0388975, 3.6511269, -1.7035588, 1.5923096, -5.6312070, 5.3546858
2: -6.7433052, 2.7617259, -2.4423940, 1.5007048, -8.2440100, 5.2041197
3: -5.9124675, 2.9107666, -2.2464468, 1.3815364, -7.2940040, 5.1572132
4: -6.1932940, 3.8548326, -2.4962273, 1.7174634, -7.9107575, 6.3510599
5: -4.6728020, 4.1081657, -1.9467959, 1.7799761, -6.4527779, 6.0549617
6: -4.8909769, 4.1707125, -2.0029566, 1.7588335, -6.6498103, 6.1736689
7: -5.6431022, 4.1655025, -2.2908437, 1.8423295, -7.4854317, 6.4563465
8: -6.4577360, 3.8242424, -2.5968261, 1.7474377, -8.2051735, 6.4210682
9: -4.5612731, 5.2246933, -1.9660535, 2.1876166, -6.7488899, 7.1907468

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 30

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6299404
time: 2.51 seconds

## Relational analysis of NS_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6324578
time: 5.24 seconds

## BFS NS instance: NS_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.2258110, 4.0003090, -5.7575884, 4.3945508, -9.6203613, 9.7578974
1: -4.0617042, 3.6687789, -4.2299485, 4.0489388, -8.1106434, 7.8987274
2: -6.8062730, 2.7964144, -7.6227016, 2.8517237, -9.6579971, 10.4191160
3: -5.9453349, 2.9258800, -6.6704755, 3.1960638, -9.1413984, 9.5963554
4: -6.2242832, 3.8764381, -6.9428921, 4.2515326, -10.4758158, 10.8193302
5: -4.6994996, 4.1530495, -5.2222462, 4.5900369, -9.2895365, 9.3752956
6: -4.9172645, 4.1935554, -5.4462862, 4.6091986, -9.5264626, 9.6398411
7: -5.6777139, 4.1865282, -6.3334379, 4.5891066, -10.2668209, 10.5199661
8: -6.4980412, 3.8435464, -7.1542482, 4.2342615, -10.7323027, 10.9977951
9: -4.5820532, 5.2544990, -5.0722699, 5.8165703, -10.3986235, 10.3267689

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 161

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_A2_B2_A1_A1

### Relational analysis result of NS_B1_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303956, upper bound: 7.6290450
time: 3.59 seconds

## Relational analysis of NS_B1_A2_A2_B2_A1_A2

### Relational analysis result of NS_B1_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289787, upper bound: 7.6289850
time: 2.74 seconds

## BFS NS instance: NS_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.2124267, 4.7214203, -5.7575884, 4.3945508, -10.6069775, 10.4790087
1: -4.8502493, 4.3227358, -4.2299485, 4.0489388, -8.8991880, 8.5526848
2: -8.1273041, 3.2157297, -7.6227016, 2.8517237, -10.9790277, 10.8384314
3: -7.1248002, 3.4151611, -6.6704755, 3.1960638, -10.3208637, 10.0856361
4: -7.3889132, 4.5644493, -6.9428921, 4.2515326, -11.6404457, 11.5073414
5: -5.5602236, 4.8745117, -5.2222462, 4.5900369, -10.1502609, 10.0967579
6: -5.8263245, 4.9686246, -5.4462862, 4.6091986, -10.4355230, 10.4149113
7: -6.7332931, 4.9369349, -6.3334379, 4.5891066, -11.3223991, 11.2703724
8: -7.7301259, 4.5107884, -7.1542482, 4.2342615, -11.9643879, 11.6650372
9: -5.4027495, 6.2235818, -5.0722699, 5.8165703, -11.2193203, 11.2958517

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 230

## Relational analysis of NS_B1_A2_A2_B2_A2_A1

### Relational analysis result of NS_B1_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324069, upper bound: 7.6324195
time: 3.09 seconds

## Relational analysis of NS_B1_A2_A2_B2_A2_A2

### Relational analysis result of NS_B1_A2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6324208
time: 2.85 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -2.2380745, 1.8050854, -2.2355657, 1.7990253, -4.0370998, 4.0406513
1: -1.7980882, 1.6848613, -1.7956465, 1.6860437, -3.4841318, 3.4805079
2: -2.6492724, 1.5403659, -2.6386285, 1.5432963, -4.1925688, 4.1789942
3: -2.3990428, 1.4490991, -2.3953915, 1.4480960, -3.8471389, 3.8444905
4: -2.6521759, 1.8108498, -2.6439750, 1.8106759, -4.4628515, 4.4548249
5: -2.0623641, 1.8858886, -2.0660846, 1.8807189, -3.9430830, 3.9519732
6: -2.1315165, 1.8590215, -2.1259298, 1.8584037, -3.9899201, 3.9849515
7: -2.4316661, 1.9428017, -2.4252632, 1.9449842, -4.3766503, 4.3680649
8: -2.7659321, 1.8354522, -2.7569976, 1.8393042, -4.6052361, 4.5924497
9: -2.0822058, 2.3160677, -2.0834792, 2.3034928, -4.3856983, 4.3995466

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
time: 2.28 seconds

## Relational analysis of NS_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
time: 2.34 seconds

## BFS NS instance: NS_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -5.6185951, 4.2716193, -2.2355657, 1.7990253, -7.4176207, 6.5071850
1: -4.1213198, 3.9566376, -1.7956465, 1.6860437, -5.8073635, 5.7522840
2: -7.4328771, 2.7999535, -2.6386285, 1.5432963, -8.9761734, 5.4385819
3: -6.1725292, 3.1276188, -2.3953915, 1.4480960, -7.6206255, 5.5230103
4: -6.7716045, 4.1562223, -2.6439750, 1.8106759, -8.5822802, 6.8001976
5: -5.0997190, 4.4419203, -2.0660846, 1.8807189, -6.9804382, 6.5080051
6: -5.3159370, 4.4713960, -2.1259298, 1.8584037, -7.1743407, 6.5973258
7: -6.1811018, 4.4241567, -2.4252632, 1.9449842, -8.1260862, 6.8494196
8: -6.9815183, 4.0893936, -2.7569976, 1.8393042, -8.8208227, 6.8463912
9: -4.9549112, 5.5824480, -2.0834792, 2.3034928, -7.2584038, 7.6659269

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
time: 2.40 seconds

## Relational analysis of NS_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
time: 3.64 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -2.2380745, 1.8050854, -5.6204772, 4.2910242, -6.5290985, 7.4255629
1: -1.7980882, 1.6848613, -4.1353722, 3.9699244, -5.7680125, 5.8202333
2: -2.6492724, 1.5403659, -7.4287090, 2.8142042, -5.4634767, 8.9690752
3: -2.3990428, 1.4490991, -6.4994969, 3.1324167, -5.5314598, 7.9485960
4: -2.6521759, 1.8108498, -6.7710991, 4.1788130, -6.8309889, 8.5819492
5: -2.0623641, 1.8858886, -5.1077037, 4.4831533, -6.5455174, 6.9935923
6: -2.1315165, 1.8590215, -5.3151417, 4.5160017, -6.6475182, 7.1741633
7: -2.4316661, 1.9428017, -6.1839952, 4.4906549, -6.9223213, 8.1267967
8: -2.7659321, 1.8354522, -6.9841442, 4.1458650, -6.9117970, 8.8195963
9: -2.0822058, 2.3160677, -4.9601774, 5.6757460, -7.7579517, 7.2762451

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304620, upper bound: 7.6283727
time: 3.34 seconds

## Relational analysis of NS_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320005, upper bound: 7.6319276
time: 3.40 seconds

## BFS NS instance: NS_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -5.6185951, 4.2716193, -5.6204772, 4.2910242, -9.9096193, 9.8920965
1: -4.1213198, 3.9566376, -4.1353722, 3.9699244, -8.0912437, 8.0920095
2: -7.4328771, 2.7999535, -7.4287090, 2.8142042, -10.2470818, 10.2286625
3: -6.1725292, 3.1276188, -6.4994969, 3.1324167, -9.3049459, 9.6271152
4: -6.7716045, 4.1562223, -6.7710991, 4.1788130, -10.9504175, 10.9273214
5: -5.0997190, 4.4419203, -5.1077037, 4.4831533, -9.5828724, 9.5496235
6: -5.3159370, 4.4713960, -5.3151417, 4.5160017, -9.8319387, 9.7865372
7: -6.1811018, 4.4241567, -6.1839952, 4.4906549, -10.6717567, 10.6081524
8: -6.9815183, 4.0893936, -6.9841442, 4.1458650, -11.1273832, 11.0735378
9: -4.9549112, 5.5824480, -4.9601774, 5.6757460, -10.6306572, 10.5426254

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323954, upper bound: 7.6323387
time: 3.23 seconds

## Relational analysis of NS_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323964, upper bound: 7.6323371
time: 13.44 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -2.3338535, 1.8806324, -2.6196442, 2.0933833, -4.4272366, 4.5002766
1: -1.8705406, 1.7552400, -2.0746675, 1.9575135, -3.8280540, 3.8299074
2: -2.7982168, 1.5723163, -3.2131815, 1.6855036, -4.4837203, 4.7854977
3: -2.5174484, 1.5006043, -2.8486047, 1.6480401, -4.1654882, 4.3492088
4: -2.7822888, 1.8823733, -3.1411972, 2.0911605, -4.8734493, 5.0235705
5: -2.1522994, 1.9604743, -2.4237320, 2.1737525, -4.3260517, 4.3842063
6: -2.2313132, 1.9377041, -2.5094218, 2.1663046, -4.3976178, 4.4471259
7: -2.5428867, 2.0178661, -2.8701410, 2.2375689, -4.7804556, 4.8880072
8: -2.8974190, 1.9023521, -3.2649107, 2.1014581, -4.9988770, 5.1672630
9: -2.1729269, 2.4178085, -2.4290111, 2.6969142, -4.8698411, 4.8468199

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B1_A1_A1

### Relational analysis result of NS_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
time: 2.93 seconds

## Relational analysis of NS_B2_A1_B2_B1_A1_A2

### Relational analysis result of NS_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
time: 3.01 seconds

## BFS NS instance: NS_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -5.7596278, 4.3945508, -2.6196442, 2.0933833, -7.8530111, 7.0141950
1: -4.2299485, 4.0584455, -2.0746675, 1.9575135, -6.1874619, 6.1331129
2: -7.6227016, 2.8566599, -3.2131815, 1.6855036, -9.3082056, 6.0698414
3: -6.6704755, 3.2000456, -2.8486047, 1.6480401, -8.3185158, 6.0486503
4: -6.9428921, 4.2715836, -3.1411972, 2.0911605, -9.0340528, 7.4127808
5: -5.2225118, 4.5900369, -2.4237320, 2.1737525, -7.3962641, 7.0137691
6: -5.4462862, 4.6206875, -2.5094218, 2.1663046, -7.6125908, 7.1301093
7: -6.3334379, 4.5902991, -2.8701410, 2.2375689, -8.5710068, 7.4604402
8: -7.1581602, 4.2342615, -3.2649107, 2.1014581, -9.2596188, 7.4991722
9: -5.0722699, 5.8229094, -2.4290111, 2.6969142, -7.7691841, 8.2519207

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
time: 1.95 seconds

## Relational analysis of NS_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
time: 2.60 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2.3338535, 1.8806324, -6.1708174, 4.6784267, -7.0122805, 8.0514498
1: -1.8705406, 1.7552400, -4.8206434, 4.2905436, -6.1610842, 6.5758834
2: -2.7982168, 1.5723163, -8.0471668, 3.1853733, -5.9835901, 9.6194830
3: -2.5174484, 1.5006043, -7.0704432, 3.3784189, -5.8958673, 8.5710478
4: -2.7822888, 1.8823733, -7.3469925, 4.5044041, -7.2866926, 9.2293663
5: -2.1522994, 1.9604743, -5.5247989, 4.8391924, -6.9914918, 7.4852734
6: -2.2313132, 1.9377041, -5.7866602, 4.9247112, -7.1560245, 7.7243643
7: -2.5428867, 2.0178661, -6.6892972, 4.9057126, -7.4485993, 8.7071629
8: -2.8974190, 1.9023521, -7.6365604, 4.4801340, -7.3775530, 9.5389128
9: -2.1729269, 2.4178085, -5.3680449, 6.1882858, -8.3612127, 7.7858534

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304678, upper bound: 7.6261537
time: 3.78 seconds

## Relational analysis of NS_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320278, upper bound: 7.6320166
time: 3.18 seconds

## BFS NS instance: NS_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -5.7575884, 4.3945508, -6.1708174, 4.6784267, -10.4360151, 10.5653687
1: -4.2299485, 4.0489388, -4.8206434, 4.2905436, -8.5204926, 8.8695822
2: -7.6227016, 2.8517237, -8.0471668, 3.1853733, -10.8080750, 10.8988905
3: -6.6704755, 3.1960638, -7.0704432, 3.3784189, -10.0488949, 10.2665071
4: -6.9428921, 4.2515326, -7.3469925, 4.5044041, -11.4472961, 11.5985250
5: -5.2222462, 4.5900369, -5.5247989, 4.8391924, -10.0614386, 10.1148357
6: -5.4462862, 4.6091986, -5.7866602, 4.9247112, -10.3709974, 10.3958588
7: -6.3334379, 4.5891066, -6.6892972, 4.9057126, -11.2391510, 11.2784042
8: -7.1542482, 4.2342615, -7.6365604, 4.4801340, -11.6343822, 11.8708220
9: -5.0722699, 5.8165703, -5.3680449, 6.1882858, -11.2605553, 11.1846151

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 230

## Relational analysis of NS_B2_A1_B2_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324195, upper bound: 7.6324074
time: 3.57 seconds

## Relational analysis of NS_B2_A1_B2_B2_A2_B2

### Relational analysis result of NS_B2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324201, upper bound: 7.6324095
time: 2.85 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -3.1187835, 2.4610908, -2.2355657, 1.7990253, -4.9178085, 4.6966562
1: -2.4174776, 2.2936308, -1.7956465, 1.6860437, -4.1035213, 4.0892773
2: -3.9226675, 1.8811945, -2.6386285, 1.5432963, -5.4659638, 4.5198231
3: -3.4405944, 1.8972802, -2.3953915, 1.4480960, -4.8886905, 4.2926717
4: -3.7432592, 2.4423981, -2.6439750, 1.8106759, -5.5539351, 5.0863733
5: -2.8696542, 2.5558004, -2.0660846, 1.8807189, -4.7503729, 4.6218853
6: -2.9738081, 2.5542490, -2.1259298, 1.8584037, -4.8322115, 4.6801786
7: -3.4211364, 2.6120365, -2.4252632, 1.9449842, -5.3661203, 5.0372996
8: -3.8898296, 2.4390495, -2.7569976, 1.8393042, -5.7291336, 5.1960468
9: -2.8492255, 3.1873357, -2.0834792, 2.3034928, -5.1527185, 5.2708149

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_B1_A1_A1

### Relational analysis result of NS_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
time: 2.86 seconds

## Relational analysis of NS_B2_A2_B1_B1_A1_A2

### Relational analysis result of NS_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
time: 2.57 seconds

## BFS NS instance: NS_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -6.7113256, 5.0800357, -2.2355657, 1.7990253, -8.5103512, 7.3156013
1: -5.2438636, 4.6465702, -1.7956465, 1.6860437, -6.9299073, 6.4422169
2: -8.8025398, 3.4226661, -2.6386285, 1.5432963, -10.3458366, 6.0612946
3: -7.7096291, 3.6651051, -2.3953915, 1.4480960, -9.1577253, 6.0604963
4: -7.9684415, 4.9040861, -2.6439750, 1.8106759, -9.7791176, 7.5480614
5: -5.9831090, 5.2374344, -2.0660846, 1.8807189, -7.8638277, 7.3035192
6: -6.2747841, 5.3573856, -2.1259298, 1.8584037, -8.1331882, 7.4833155
7: -7.2645969, 5.3097229, -2.4252632, 1.9449842, -9.2095814, 7.7349863
8: -8.3465309, 4.8438973, -2.7569976, 1.8393042, -10.1858349, 7.6008949
9: -5.8098416, 6.7254162, -2.0834792, 2.3034928, -8.1133347, 8.8088951

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_B1_A2_A1

### Relational analysis result of NS_B2_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
time: 3.92 seconds

## Relational analysis of NS_B2_A2_B1_B1_A2_A2

### Relational analysis result of NS_B2_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
time: 2.75 seconds

## BFS NS instance: NS_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -3.1187835, 2.4610908, -5.6204772, 4.2910242, -7.4098077, 8.0815678
1: -2.4174776, 2.2936308, -4.1353722, 3.9699244, -6.3874021, 6.4290028
2: -3.9226675, 1.8811945, -7.4287090, 2.8142042, -6.7368717, 9.3099031
3: -3.4405944, 1.8972802, -6.4994969, 3.1324167, -6.5730114, 8.3967772
4: -3.7432592, 2.4423981, -6.7710991, 4.1788130, -7.9220724, 9.2134972
5: -2.8696542, 2.5558004, -5.1077037, 4.4831533, -7.3528075, 7.6635041
6: -2.9738081, 2.5542490, -5.3151417, 4.5160017, -7.4898100, 7.8693905
7: -3.4211364, 2.6120365, -6.1839952, 4.4906549, -7.9117913, 8.7960320
8: -3.8898296, 2.4390495, -6.9841442, 4.1458650, -8.0356941, 9.4231939
9: -2.8492255, 3.1873357, -4.9601774, 5.6757460, -8.5249710, 8.1475134

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304162, upper bound: 7.6283728
time: 5.96 seconds

## Relational analysis of NS_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_B2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319897, upper bound: 7.6319274
time: 2.64 seconds

## BFS NS instance: NS_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -6.7113256, 5.0795760, -5.6204772, 4.2910242, -11.0023499, 10.7000532
1: -5.2438636, 4.6446295, -4.1353722, 3.9699244, -9.2137880, 8.7800016
2: -8.8019905, 3.4226661, -7.4287090, 2.8142042, -11.6161947, 10.8513756
3: -7.7068291, 3.6651051, -6.4994969, 3.1324167, -10.8392458, 10.1646023
4: -7.9671807, 4.9040861, -6.7710991, 4.1788130, -12.1459942, 11.6751852
5: -5.9821901, 5.2374344, -5.1077037, 4.4831533, -10.4653435, 10.3451385
6: -6.2747841, 5.3565526, -5.3151417, 4.5160017, -10.7907858, 10.6716938
7: -7.2645969, 5.3065777, -6.1839952, 4.4906549, -11.7552519, 11.4905729
8: -8.3465309, 4.8385329, -6.9841442, 4.1458650, -12.4923954, 11.8226776
9: -5.8058958, 6.7254162, -4.9601774, 5.6757460, -11.4816418, 11.6855936

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B1_B2_A2_A1

### Relational analysis result of NS_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308999, upper bound: 7.6297265
time: 4.60 seconds

## Relational analysis of NS_B2_A2_B1_B2_A2_A2

### Relational analysis result of NS_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6297871, upper bound: 7.6296657
time: 3.05 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -3.2403135, 2.5479503, -2.6196442, 2.0933833, -5.3336968, 5.1675944
1: -2.4995601, 2.3735836, -2.0746675, 1.9575135, -4.4570737, 4.4482508
2: -4.0877514, 1.9280030, -3.2131815, 1.6855036, -5.7732549, 5.1411843
3: -3.5869420, 1.9565104, -2.8486047, 1.6480401, -5.2349820, 4.8051152
4: -3.8888640, 2.5249925, -3.1411972, 2.0911605, -5.9800243, 5.6661897
5: -2.9752712, 2.6450162, -2.4237320, 2.1737525, -5.1490240, 5.0687485
6: -3.0867250, 2.6473107, -2.5094218, 2.1663046, -5.2530298, 5.1567326
7: -3.5516436, 2.7015889, -2.8701410, 2.2375689, -5.7892122, 5.5717297
8: -4.0402813, 2.5194533, -3.2649107, 2.1014581, -6.1417394, 5.7843637
9: -2.9498816, 3.3069129, -2.4290111, 2.6969142, -5.6467957, 5.7359238

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B2_B1_A1_A1

### Relational analysis result of NS_B2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307397
time: 2.62 seconds

## Relational analysis of NS_B2_A2_B2_B1_A1_A2

### Relational analysis result of NS_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324249
time: 5.66 seconds

## BFS NS instance: NS_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -6.8564267, 5.1819391, -2.6196442, 2.0933833, -8.9498100, 7.8015833
1: -5.3554454, 4.7429519, -2.0746675, 1.9575135, -7.3129587, 6.8176193
2: -8.9886723, 3.4785094, -3.2131815, 1.6855036, -10.6741762, 6.6916909
3: -7.8770094, 3.7373860, -2.8486047, 1.6480401, -9.5250492, 6.5859909
4: -8.1349955, 5.0116653, -3.1411972, 2.0911605, -10.2261562, 8.1528625
5: -6.1032343, 5.3428826, -2.4237320, 2.1737525, -8.2769871, 7.7666149
6: -6.4196472, 5.4665518, -2.5094218, 2.1663046, -8.5859518, 7.9759736
7: -7.4192195, 5.4149461, -2.8701410, 2.2375689, -9.6567879, 8.2850876
8: -8.5203619, 4.9379287, -3.2649107, 2.1014581, -10.6218204, 8.2028389
9: -5.9301400, 6.8654613, -2.4290111, 2.6969142, -8.6270542, 9.2944727

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B2_B1_A2_A1

### Relational analysis result of NS_B2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307398
time: 5.01 seconds

## Relational analysis of NS_B2_A2_B2_B1_A2_A2

### Relational analysis result of NS_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324248
time: 3.50 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -3.2403135, 2.5479503, -6.1708174, 4.6784267, -7.9187403, 8.7187672
1: -2.4995601, 2.3735836, -4.8206434, 4.2905436, -6.7901039, 7.1942272
2: -4.0877514, 1.9280030, -8.0471668, 3.1853733, -7.2731247, 9.9751701
3: -3.5869420, 1.9565104, -7.0704432, 3.3784189, -6.9653606, 9.0269537
4: -3.8888640, 2.5249925, -7.3469925, 4.5044041, -8.3932686, 9.8719845
5: -2.9752712, 2.6450162, -5.5247989, 4.8391924, -7.8144636, 8.1698151
6: -3.0867250, 2.6473107, -5.7866602, 4.9247112, -8.0114365, 8.4339714
7: -3.5516436, 2.7015889, -6.6892972, 4.9057126, -8.4573565, 9.3908863
8: -4.0402813, 2.5194533, -7.6365604, 4.4801340, -8.5204153, 10.1560135
9: -2.9498816, 3.3069129, -5.3680449, 6.1882858, -9.1381674, 8.6749573

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304214, upper bound: 7.6261536
time: 5.97 seconds

## Relational analysis of NS_B2_A2_B2_B2_A1_B2

### Relational analysis result of NS_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320163, upper bound: 7.6320169
time: 2.70 seconds

## BFS NS instance: NS_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -6.8564267, 5.1819391, -6.1708174, 4.6784267, -11.5348530, 11.3527565
1: -5.3554454, 4.7429519, -4.8206434, 4.2905436, -9.6459885, 9.5635948
2: -8.9886723, 3.4785094, -8.0471668, 3.1853733, -12.1740456, 11.5256767
3: -7.8770094, 3.7373860, -7.0704432, 3.3784189, -11.2554283, 10.8078289
4: -8.1349955, 5.0116653, -7.3469925, 4.5044041, -12.6393995, 12.3586578
5: -6.1032343, 5.3428826, -5.5247989, 4.8391924, -10.9424267, 10.8676815
6: -6.4196472, 5.4665518, -5.7866602, 4.9247112, -11.3443584, 11.2532120
7: -7.4192195, 5.4149461, -6.6892972, 4.9057126, -12.3249321, 12.1042433
8: -8.5203619, 4.9379287, -7.6365604, 4.4801340, -13.0004959, 12.5744896
9: -5.9301400, 6.8654613, -5.3680449, 6.1882858, -12.1184254, 12.2335062

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_B2_B2_A2_A1

### Relational analysis result of NS_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309514, upper bound: 7.6298900
time: 4.96 seconds

## Relational analysis of NS_B2_A2_B2_B2_A2_A2

### Relational analysis result of NS_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6298321, upper bound: 7.6298323
time: 4.51 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 10.43 seconds
NS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6300020, upper bound: 7.5860490
NS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6302489, upper bound: 7.5860504
NS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6305727, upper bound: 7.6315649
NS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6305728, upper bound: 7.6326769
NS_B1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6290859, upper bound: 7.4743615
NS_B1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6289733, upper bound: 7.4419433
NS_B1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6316419, upper bound: 7.6309815
NS_B1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6317932, upper bound: 7.6314472
NS_B1_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.4878332, upper bound: 7.6293744
NS_B1_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.4224551, upper bound: 7.6292976
NS_B1_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6299327
NS_B1_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304189, upper bound: 7.6324261
NS_B1_A1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6302952, upper bound: 7.6290127
NS_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6288282, upper bound: 7.6289555
NS_B1_A1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6323388, upper bound: 7.6323961
NS_B1_A1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6323366, upper bound: 7.6323964
NS_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6300528, upper bound: 7.5861040
NS_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6303786, upper bound: 7.5861222
NS_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6303598, upper bound: 7.6315650
NS_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6303594, upper bound: 7.6326976
NS_B1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6293385, upper bound: 7.4749466
NS_B1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6292978, upper bound: 7.4420164
NS_B1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6317450, upper bound: 7.6309989
NS_B1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6319177, upper bound: 7.6314989
NS_B1_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.4716106, upper bound: 7.6294428
NS_B1_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.4404536, upper bound: 7.6294079
NS_B1_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6299404
NS_B1_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304622, upper bound: 7.6324578
NS_B1_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6303956, upper bound: 7.6290450
NS_B1_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6289787, upper bound: 7.6289850
NS_B1_A2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6324069, upper bound: 7.6324195
NS_B1_A2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6324090, upper bound: 7.6324208
NS_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
NS_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
NS_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6281443, upper bound: 7.6307164
NS_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320192, upper bound: 7.6323443
NS_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304620, upper bound: 7.6283727
NS_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320005, upper bound: 7.6319276
NS_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6323954, upper bound: 7.6323387
NS_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6323964, upper bound: 7.6323371
NS_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
NS_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
NS_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6281815, upper bound: 7.6307377
NS_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320428, upper bound: 7.6324226
NS_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304678, upper bound: 7.6261537
NS_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320278, upper bound: 7.6320166
NS_B2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6324195, upper bound: 7.6324074
NS_B2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6324201, upper bound: 7.6324095
NS_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
NS_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
NS_B2_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6261239, upper bound: 7.6307169
NS_B2_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320108, upper bound: 7.6323470
NS_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304162, upper bound: 7.6283728
NS_B2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6319897, upper bound: 7.6319274
NS_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6308999, upper bound: 7.6297265
NS_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6297871, upper bound: 7.6296657
NS_B2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307397
NS_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324249
NS_B2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6261728, upper bound: 7.6307398
NS_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320342, upper bound: 7.6324248
NS_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6304214, upper bound: 7.6261536
NS_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6320163, upper bound: 7.6320169
NS_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6309514, upper bound: 7.6298900
NS_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 10.43
Output dim: 2, lower bound: -7.6298321, upper bound: 7.6298323

## BFS NS instance: NS_B1_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.1658729, 0.1789215, -0.0850829, 0.0905871, -0.2564601, 0.2640044
1: -0.2131706, 0.2248433, -0.1200459, 0.1227683, -0.3359389, 0.3448892
2: 0.5837186, 1.0349226, 0.7297691, 1.0348905, -0.4511719, 0.3051535
3: -0.0843362, 0.2963609, -0.0192269, 0.1865335, -0.2708696, 0.3155877
4: -0.2301640, 0.2246756, -0.1336302, 0.1527331, -0.3828971, 0.3583059
5: -0.2078561, 0.2280055, -0.1110839, 0.1204022, -0.3282583, 0.3390895
6: -0.1675590, 0.2393126, -0.1000570, 0.1163515, -0.2839105, 0.3393696
7: -0.1965176, 0.2550813, -0.1263874, 0.1421578, -0.3386753, 0.3814687
8: -0.2247507, 0.3218389, -0.1157629, 0.2033019, -0.4280526, 0.4376018
9: -0.2367792, 0.2093333, -0.1304644, 0.1094684, -0.3462476, 0.3397977

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2970848, upper bound: 7.3341162
time: 3.82 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2587349, upper bound: 7.0825420
time: 2.20 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.2564127, 0.3025258, -0.0942038, 0.1006119, -0.3570246, 0.3967296
1: -0.3072854, 0.3324686, -0.1306143, 0.1343081, -0.4415935, 0.4630829
2: 0.4355907, 1.0384488, 0.7131944, 1.0348853, -0.5992945, 0.3252544
3: -0.1710162, 0.3982040, -0.0241034, 0.1989651, -0.3699814, 0.4223075
4: -0.3371472, 0.3211084, -0.1444858, 0.1608976, -0.4980448, 0.4655942
5: -0.3034618, 0.3302253, -0.1220653, 0.1326104, -0.4360722, 0.4522907
6: -0.2486188, 0.3662626, -0.1077176, 0.1302430, -0.3788618, 0.4739802
7: -0.2927764, 0.3730482, -0.1343461, 0.1548893, -0.4476657, 0.5073944
8: -0.3508220, 0.4395191, -0.1281262, 0.2167221, -0.5675441, 0.5676453
9: -0.3502190, 0.3169457, -0.1424873, 0.1208016, -0.4710205, 0.4594330

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3120376, upper bound: 7.3275298
time: 2.21 seconds

## Relational analysis of NS_B1_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2809754, upper bound: 7.0825371
time: 3.39 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1661583, 0.1792144, -1.3644453, 1.1501882, -1.3163465, 1.5436597
1: -0.2134795, 0.2251934, -1.1318805, 1.0669785, -1.2804580, 1.3570739
2: 0.5832343, 1.0376374, -1.2343626, 1.2828192, -0.6995849, 2.2719998
3: -0.0845616, 0.2967353, -1.3378156, 0.9844181, -1.0689797, 1.6345510
4: -0.2305093, 0.2249143, -1.5969075, 1.1730683, -1.4035776, 1.8218218
5: -0.2081773, 0.2283639, -1.2479331, 1.2082707, -1.4164480, 1.4762970
6: -0.1677829, 0.2397378, -1.2479024, 1.1776177, -1.3454006, 1.4876403
7: -0.1967502, 0.2554704, -1.4493136, 1.2553008, -1.4520509, 1.7047840
8: -0.2251131, 0.3222454, -1.5698447, 1.2372620, -1.4623752, 1.8920901
9: -0.2371411, 0.2096647, -1.2857995, 1.4212538, -1.6583949, 1.4954642

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5590883, upper bound: 7.6301223
time: 2.81 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5590860, upper bound: 7.6304251
time: 3.14 seconds

## BFS NS instance: NS_B1_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.4109960, 1.1827054, -1.3644453, 1.1501882, -2.5611842, 2.5471506
1: -1.1675264, 1.0992594, -1.1318805, 1.0669785, -2.2345047, 2.2311399
2: -1.3038876, 1.2964505, -1.2343626, 1.2828192, -2.5867066, 2.5308132
3: -1.3938557, 1.0092511, -1.3378156, 0.9844181, -2.3782737, 2.3470669
4: -1.6512403, 1.2071908, -1.5969075, 1.1730683, -2.8243086, 2.8040981
5: -1.2942102, 1.2376652, -1.2479331, 1.2082707, -2.5024810, 2.4855983
6: -1.2916391, 1.2119440, -1.2479024, 1.1776177, -2.4692569, 2.4598465
7: -1.4991890, 1.2927310, -1.4493136, 1.2553008, -2.7544899, 2.7420447
8: -1.6307559, 1.2713658, -1.5698447, 1.2372620, -2.8680179, 2.8412104
9: -1.3276224, 1.4660439, -1.2857995, 1.4212538, -2.7488761, 2.7518435

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_B1_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5590883, upper bound: 7.6318126
time: 2.83 seconds

## Relational analysis of NS_B1_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_B1_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5590860, upper bound: 7.6321972
time: 4.85 seconds

## BFS NS instance: NS_B1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.3450565, 0.4039487, -0.3738203, 0.4331812, -0.7782377, 0.7777690
1: -0.3901505, 0.4158610, -0.4142024, 0.4408349, -0.8309854, 0.8300633
2: 0.2976028, 1.0632744, 0.2551078, 1.0682517, -0.7706490, 0.8081665
3: -0.2410245, 0.4688961, -0.2630281, 0.4882739, -0.7292985, 0.7319242
4: -0.4241700, 0.4219686, -0.4473135, 0.4532401, -0.8774102, 0.8692821
5: -0.3790457, 0.4354362, -0.4048980, 0.4637878, -0.8428335, 0.8403343
6: -0.3279573, 0.4558863, -0.3519856, 0.4813309, -0.8092883, 0.8078719
7: -0.3804221, 0.4763546, -0.4067290, 0.5066184, -0.8870405, 0.8830836
8: -0.4429621, 0.5319207, -0.4667529, 0.5598958, -1.0028579, 0.9986736
9: -0.4362593, 0.4270324, -0.4590568, 0.4622113, -0.8984706, 0.8860892

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_B2_B1_B1_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2863381, upper bound: 7.2582699
time: 3.29 seconds

## Relational analysis of NS_B1_A1_A1_B2_B1_B1_B2

### Relational analysis result of NS_B1_A1_A1_B2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2682489, upper bound: 7.1070431
time: 2.42 seconds

## BFS NS instance: NS_B1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.3693123, 0.4308851, -0.4784258, 0.5496858, -0.9189981, 0.9093109
1: -0.4121934, 0.4391969, -0.5152997, 0.5395699, -0.9517633, 0.9544966
2: 0.2590156, 1.0694418, 0.0878532, 1.0939983, -0.8349828, 0.9815886
3: -0.2599484, 0.4875208, -0.3428099, 0.5690678, -0.8290161, 0.8303306
4: -0.4468151, 0.4507382, -0.5496060, 0.5739964, -1.0208114, 1.0003442
5: -0.4020609, 0.4634911, -0.5098730, 0.5869027, -0.9889637, 0.9733642
6: -0.3497947, 0.4791698, -0.4516047, 0.5790055, -0.9288002, 0.9307745
7: -0.4035629, 0.5056469, -0.5098636, 0.6309913, -1.0345541, 1.0155104
8: -0.4666375, 0.5565438, -0.5712078, 0.6741465, -1.1407840, 1.1277516
9: -0.4585651, 0.4595352, -0.5543026, 0.6052815, -1.0638466, 1.0138378

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_B2_B1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2621290, upper bound: 7.2398641
time: 2.50 seconds

## Relational analysis of NS_B1_A1_A1_B2_B1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2366384, upper bound: 7.0286966
time: 2.84 seconds

## BFS NS instance: NS_B1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -1.2533991, 1.0660272, -2.2083836, 1.7794564, -3.0328555, 3.2744107
1: -1.0487903, 0.9964671, -1.7747747, 1.6562681, -2.7050586, 2.7712417
2: -1.0541703, 1.2541554, -2.5932770, 1.5251312, -2.5793014, 3.8474324
3: -1.2013261, 0.9288639, -2.3630216, 1.4293001, -2.6306262, 3.2918856
4: -1.4595424, 1.0981048, -2.6097741, 1.7835920, -3.2431345, 3.7078791
5: -1.1494367, 1.1277304, -2.0357776, 1.8487598, -2.9981966, 3.1635079
6: -1.1389773, 1.0945668, -2.1054597, 1.8350239, -2.9740012, 3.2000265
7: -1.3229729, 1.1730320, -2.3990641, 1.9096737, -3.2326465, 3.5720961
8: -1.4138058, 1.1687766, -2.7230072, 1.8111256, -3.2249315, 3.8917837
9: -1.1878093, 1.3060873, -2.0496392, 2.2822001, -3.4700093, 3.3557265

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 240
type: B, layer: 1, pos: 240
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 58
type: B, layer: 1, pos: 58
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_B1_A1_A1_B2_B2_B1_A1

### Relational analysis result of NS_B1_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5159742, upper bound: 7.6117656
time: 3.36 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2_B1_A2

### Relational analysis result of NS_B1_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.5159742, upper bound: 7.6309813
time: 2.87 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 6.01 + 596.39 = 602.40 seconds
