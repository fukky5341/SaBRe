## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1888011651


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3603992, 3.3603988)
1: (-11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0151329, 3.0151334)
2: (-10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6060905, 3.6060905)
3: (-5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4481053, 2.4481056)
4: (-11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5820861, 2.5820856)
5: (6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1325693, 2.1325696)
6: (-8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8638582, 2.8638577)
7: (-17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1436224, 3.1436229)
8: (-6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6549873, 2.6549873)
9: (-4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3357582, 2.3357592)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.02 + 39.76 = 63.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.1923782, upper bound: 1.1923775

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5777
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5777

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1706274, upper bound: 1.1863885
time: 10.65 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923619, upper bound: 1.1923640
time: 9.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 20.14 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 20.14
Output dim: 5, lower bound: -1.1706274, upper bound: 1.1863885
NS_A2, status: Status.UNKNOWN, split count: 1, time: 20.14
Output dim: 5, lower bound: -1.1923619, upper bound: 1.1923640

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.0934525, -5.3854027, -9.0934563, -5.3853974, -3.3582730, 3.3593130
1: -11.2401667, -7.5092626, -11.2401724, -7.5092559, -3.0121431, 2.9965363
2: -10.3444214, -6.3544092, -10.3444252, -6.3544049, -3.6046190, 3.6175256
3: -5.0487928, -2.3199065, -5.0487976, -2.3199029, -2.4481478, 2.4480932
4: -11.4109106, -8.3298826, -11.4109144, -8.3298759, -2.5899878, 2.5810170
5: 6.9648223, 9.4015274, 6.9648046, 9.4015274, -2.0693636, 2.1325543
6: -8.6112547, -5.0921717, -8.6112633, -5.0921702, -2.8326130, 2.8638430
7: -17.1788960, -13.3413153, -17.1788960, -13.3413105, -3.1436119, 3.1427965
8: -6.0857363, -3.1872473, -6.0857401, -3.1872296, -2.6549692, 2.6256628
9: -4.2306385, -1.7395937, -4.2306404, -1.7395837, -2.3357468, 2.3183093

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 863

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1882765, upper bound: 1.1708965
time: 10.07 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923575, upper bound: 1.1923585
time: 14.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 45.96 seconds
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 45.96
Output dim: 5, lower bound: -1.1882765, upper bound: 1.1708965
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 45.96
Output dim: 5, lower bound: -1.1923575, upper bound: 1.1923585

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.0934467, -5.3854146, -9.0934515, -5.3854170, -3.3493004, 3.3569324
1: -11.2398643, -7.5092640, -11.2396698, -7.5092583, -3.0071201, 2.9965186
2: -10.3444157, -6.3544121, -10.3444195, -6.3544092, -3.6058416, 3.6169186
3: -5.0487862, -2.3199098, -5.0487881, -2.3199069, -2.4481344, 2.4482601
4: -11.4101000, -8.3298855, -11.4095535, -8.3298817, -2.5885706, 2.5840774
5: 6.9648333, 9.4015255, 6.9648218, 9.4015284, -2.0693517, 2.1182814
6: -8.6112432, -5.0921736, -8.6112509, -5.0921721, -2.8326006, 2.8574767
7: -17.1788864, -13.3413210, -17.1788883, -13.3413143, -3.1433573, 3.1431065
8: -6.0857334, -3.1872587, -6.0857363, -3.1872435, -2.6484518, 2.6256499
9: -4.2306366, -1.7395992, -4.2306385, -1.7395918, -2.3320999, 2.3183031

Time for backsubstitution: 23.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1873489, upper bound: 1.1923510
time: 8.49 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923496, upper bound: 1.1923506
time: 22.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 60.76 seconds
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 60.76
Output dim: 5, lower bound: -1.1873489, upper bound: 1.1923510
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 60.76
Output dim: 5, lower bound: -1.1923496, upper bound: 1.1923506

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.0801067, -5.3896275, -9.0878515, -5.3872457, -3.3448143, 3.3570888
1: -11.2327681, -7.5143147, -11.2364979, -7.5113649, -2.9636264, 2.9537215
2: -10.3408508, -6.3596578, -10.3429298, -6.3566847, -3.5940418, 3.6040907
3: -5.0467038, -2.3295274, -5.0478706, -2.3239269, -2.4398150, 2.4350977
4: -11.4020653, -8.3309460, -11.4061642, -8.3303823, -2.5658951, 2.5665655
5: 6.9762230, 9.3975830, 6.9695840, 9.3998384, -2.0587416, 2.1121676
6: -8.6063023, -5.0953083, -8.6091614, -5.0936918, -2.8356705, 2.8612556
7: -17.1710320, -13.3429422, -17.1755543, -13.3421555, -3.1287127, 3.1333961
8: -6.0810671, -3.2015080, -6.0836954, -3.1931953, -2.6348362, 2.6067410
9: -4.2248244, -1.7442569, -4.2281303, -1.7415661, -2.3373508, 2.3240690

Time for backsubstitution: 23.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1815796, upper bound: 1.1909364
time: 8.15 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872992, upper bound: 1.1923031
time: 10.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.1152716, -5.3057442, -9.0934343, -5.3854241, -3.3689003, 3.3939900
1: -11.2861423, -7.4913177, -11.2396593, -7.5092683, -3.0512018, 3.0121193
2: -10.3582420, -6.2986345, -10.3444138, -6.3544168, -3.6372433, 3.6720829
3: -5.0859461, -2.3071585, -5.0487847, -2.3199220, -2.4863768, 2.4729023
4: -11.4237070, -8.2951117, -11.4095421, -8.3298836, -2.6185355, 2.6270180
5: 6.9531941, 9.4436226, 6.9648433, 9.4015217, -2.0805783, 2.1367505
6: -8.6353340, -5.0774994, -8.6112413, -5.0921779, -2.8706064, 2.8688459
7: -17.1992702, -13.3089848, -17.1788769, -13.3413172, -3.1721296, 3.1860723
8: -6.1708503, -3.1755772, -6.0857306, -3.1872673, -2.7315845, 2.6347167
9: -4.2524433, -1.7114730, -4.2306290, -1.7396001, -2.3625803, 2.3443913

Time for backsubstitution: 23.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1865713, upper bound: 1.1909359
time: 9.00 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923009, upper bound: 1.1923049
time: 9.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 42.60 seconds
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 42.60
Output dim: 5, lower bound: -1.1815796, upper bound: 1.1909364
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 42.60
Output dim: 5, lower bound: -1.1872992, upper bound: 1.1923031
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 42.60
Output dim: 5, lower bound: -1.1865713, upper bound: 1.1909359
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 42.60
Output dim: 5, lower bound: -1.1923009, upper bound: 1.1923049

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.0690107, -5.3912625, -9.0628490, -5.3942032, -3.3276639, 3.3308403
1: -11.2250929, -7.5167956, -11.2196894, -7.5202656, -2.9454269, 2.9347067
2: -10.3288660, -6.3643632, -10.3172522, -6.3720336, -3.5648746, 3.5744767
3: -5.0126371, -2.3335543, -4.9824610, -2.3501003, -2.3808069, 2.3664854
4: -11.3978348, -8.3363457, -11.3948298, -8.3432665, -2.5487680, 2.5526810
5: 6.9792643, 9.3801603, 6.9862461, 9.3672628, -2.0229821, 2.0756752
6: -8.5975628, -5.0994444, -8.5913515, -5.1152358, -2.8016052, 2.8394842
7: -17.1457634, -13.3472729, -17.1260471, -13.3676014, -3.0792694, 3.0805249
8: -6.0755959, -3.2032199, -6.0699806, -3.1992555, -2.6235371, 2.5901570
9: -4.2205248, -1.7599541, -4.2028017, -1.7714024, -2.3036966, 2.2808814

Time for backsubstitution: 22.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1808107, upper bound: 1.1909351
time: 9.06 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1815783, upper bound: 1.1909315
time: 9.08 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.0801039, -5.3896275, -9.0878448, -5.3872471, -3.3388987, 3.3572021
1: -11.2327652, -7.5143142, -11.2364931, -7.5113654, -2.9673834, 2.9532084
2: -10.3408499, -6.3596611, -10.3429241, -6.3566895, -3.5940351, 3.5914550
3: -5.0466971, -2.3295288, -5.0478554, -2.3239293, -2.4398093, 2.3952103
4: -11.4020653, -8.3309479, -11.4061632, -8.3303843, -2.5666275, 2.5664158
5: 6.9762244, 9.3975811, 6.9695868, 9.3998318, -2.0296438, 2.1096585
6: -8.6063004, -5.0953126, -8.6091566, -5.0936947, -2.8498454, 2.8593335
7: -17.1710243, -13.3429461, -17.1755409, -13.3421612, -3.1287069, 3.1059899
8: -6.0810652, -3.2015085, -6.0836911, -3.1931958, -2.6341329, 2.6059160
9: -4.2248220, -1.7442598, -4.2281284, -1.7415715, -2.3369493, 2.3269849

Time for backsubstitution: 22.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: A, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872978, upper bound: 1.1915238
time: 8.70 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872979, upper bound: 1.1923009
time: 9.49 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.1041508, -5.3073573, -9.0684290, -5.3923769, -3.3518457, 3.3677263
1: -11.2784271, -7.4937897, -11.2228575, -7.5181727, -3.0327802, 2.9930553
2: -10.3461838, -6.3032856, -10.3186893, -6.3697710, -3.6080647, 3.6424122
3: -5.0519032, -2.3111401, -4.9833798, -2.3460896, -2.4274173, 2.4043498
4: -11.4194593, -8.3004856, -11.3982019, -8.3427629, -2.6014366, 2.6131296
5: 6.9562006, 9.4261894, 6.9815035, 9.3689461, -2.0449069, 2.0937159
6: -8.6266384, -5.0816560, -8.5934429, -5.1137257, -2.8364930, 2.8469911
7: -17.1740417, -13.3132820, -17.1293831, -13.3667603, -3.1225986, 3.1332355
8: -6.1653242, -3.1772728, -6.0720105, -3.1933260, -2.7193499, 2.6179776
9: -4.2481818, -1.7271643, -4.2053041, -1.7694569, -2.3290043, 2.3012185

Time for backsubstitution: 23.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1858023, upper bound: 1.1909349
time: 7.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1865700, upper bound: 1.1909373
time: 11.96 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1152725, -5.3057456, -9.0934315, -5.3854251, -3.3634624, 3.3941200
1: -11.2861414, -7.4913197, -11.2396545, -7.5092716, -3.0518379, 3.0116072
2: -10.3582382, -6.2986341, -10.3444071, -6.3544216, -3.6372366, 3.6594381
3: -5.0859394, -2.3071585, -5.0487700, -2.3199253, -2.4863720, 2.4330149
4: -11.4237061, -8.2951097, -11.4095383, -8.3298836, -2.6193161, 2.6258588
5: 6.9531932, 9.4436207, 6.9648447, 9.4015141, -2.0514812, 2.1267042
6: -8.6353302, -5.0774961, -8.6112385, -5.0921774, -2.8752947, 2.8649220
7: -17.1992626, -13.3089828, -17.1788654, -13.3413200, -3.1721230, 3.1586685
8: -6.1708488, -3.1755772, -6.0857263, -3.1872673, -2.7287364, 2.6338911
9: -4.2524419, -1.7114756, -4.2306280, -1.7396054, -2.3621788, 2.3473077

Time for backsubstitution: 22.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 542

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1915219, upper bound: 1.1923015
time: 6.15 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1922996, upper bound: 1.1923012
time: 9.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 38.94 seconds
NS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1808107, upper bound: 1.1909351
NS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1815783, upper bound: 1.1909315
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1872978, upper bound: 1.1915238
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1872979, upper bound: 1.1923009
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1858023, upper bound: 1.1909349
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1865700, upper bound: 1.1909373
NS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1915219, upper bound: 1.1923015
NS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 38.94
Output dim: 5, lower bound: -1.1922996, upper bound: 1.1923012

## BFS NS instance: NS_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -9.0654593, -5.3980131, -9.0495720, -5.4074836, -3.3075585, 3.3064389
1: -11.2189503, -7.5369468, -11.1884384, -7.5548611, -2.9025712, 2.8816199
2: -10.3267841, -6.3765492, -10.3069792, -6.3969574, -3.5357094, 3.5441585
3: -5.0086160, -2.3374877, -4.9710016, -2.3584433, -2.3684511, 2.3506927
4: -11.3794050, -8.3397808, -11.3597946, -8.3669682, -2.5050368, 2.5124373
5: 6.9830165, 9.3742867, 7.0005836, 9.3563175, -2.0083857, 2.0538774
6: -8.5933180, -5.1063638, -8.5793266, -5.1283770, -2.7825832, 2.8162217
7: -17.1373138, -13.3581724, -17.1083641, -13.3879852, -3.0492048, 3.0480571
8: -6.0720081, -3.2137537, -6.0549526, -3.2185974, -2.5989547, 2.5637610
9: -4.2018490, -1.7643485, -4.1701317, -1.7952356, -2.2614050, 2.2438297

Time for backsubstitution: 23.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 542
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1808107, upper bound: 1.1859401
time: 9.32 seconds

## Relational analysis of NS_A2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1808107, upper bound: 1.1909351
time: 9.83 seconds

## BFS NS instance: NS_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -9.0690098, -5.3912683, -9.0628462, -5.3942122, -3.3211317, 3.3272462
1: -11.2250900, -7.5168128, -11.2196865, -7.5202904, -2.9116864, 2.9344356
2: -10.3288670, -6.3643713, -10.3172483, -6.3720446, -3.5568438, 3.5689454
3: -5.0126371, -2.3335576, -4.9824591, -2.3501062, -2.3781843, 2.3664815
4: -11.3978233, -8.3363476, -11.3948107, -8.3432674, -2.5487604, 2.5349550
5: 6.9792662, 9.3801594, 6.9862480, 9.3672581, -2.0160978, 2.0720329
6: -8.5975628, -5.0994501, -8.5913496, -5.1152415, -2.7984443, 2.8366556
7: -17.1457558, -13.3472748, -17.1260395, -13.3676052, -3.0779829, 3.0847454
8: -6.0755949, -3.2032270, -6.0699782, -3.1992683, -2.6190281, 2.5901484
9: -4.2205133, -1.7599553, -4.2027788, -1.7714047, -2.3036842, 2.2615976

Time for backsubstitution: 25.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 6136
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1815783, upper bound: 1.1859403
time: 13.52 seconds

## Relational analysis of NS_A2_B2_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1815783, upper bound: 1.1909314
time: 9.05 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.0668201, -5.4028940, -9.0842924, -5.3939919, -3.3145189, 3.3370819
1: -11.2014656, -7.5489144, -11.2303553, -7.5314960, -2.9143529, 2.9103537
2: -10.3305378, -6.3845530, -10.3407888, -6.3688726, -3.5636683, 3.5622916
3: -5.0352783, -2.3378510, -5.0438390, -2.3278542, -2.4240580, 2.3828695
4: -11.3670712, -8.3546038, -11.3877516, -8.3338089, -2.5263810, 2.5227327
5: 6.9905868, 9.3866282, 6.9733524, 9.3939581, -2.0078363, 2.0950265
6: -8.5942383, -5.1085644, -8.6049118, -5.1006331, -2.8265610, 2.8402786
7: -17.1534195, -13.3633060, -17.1671143, -13.3530560, -3.0962830, 3.0759096
8: -6.0659962, -3.2208176, -6.0800972, -3.2037244, -2.6076951, 2.5813441
9: -4.1920471, -1.7680809, -4.2094355, -1.7459841, -2.2997909, 2.2846918

Time for backsubstitution: 25.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 4636
type: A, layer: 1, pos: 4636
type: B, layer: 1, pos: 542
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6136
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5777

### Candidate
type: B, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1872978, upper bound: 1.1865209
time: 16.65 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1872978, upper bound: 1.1915261
time: 12.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.0801010, -5.3896370, -9.0878439, -5.3872523, -3.3352823, 3.3506532
1: -11.2327595, -7.5143385, -11.2364883, -7.5113797, -2.9641013, 2.9194665
2: -10.3408489, -6.3596721, -10.3429241, -6.3566942, -3.5884943, 3.5834980
3: -5.0466948, -2.3295317, -5.0478530, -2.3239305, -2.4398036, 2.3925869
4: -11.4020443, -8.3309507, -11.4061508, -8.3303833, -2.5489011, 2.5664084
5: 6.9762259, 9.3975754, 6.9695873, 9.3998289, -2.0281270, 2.1023154
6: -8.6063004, -5.0953178, -8.6091557, -5.0936995, -2.8470430, 2.8561234
7: -17.1710186, -13.3429537, -17.1755371, -13.3421631, -3.1329269, 3.1047044
8: -6.0810633, -3.2015200, -6.0836897, -3.1932034, -2.6341248, 2.6014085
9: -4.2248020, -1.7442628, -4.2281170, -1.7415731, -2.3176651, 2.3269734

Time for backsubstitution: 25.60 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 63.78 + 547.60 = 611.37 seconds
