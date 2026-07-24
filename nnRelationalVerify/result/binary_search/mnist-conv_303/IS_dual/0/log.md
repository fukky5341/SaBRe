## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.96581779658
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.2881069, -6.2814474, -9.2881069, -6.2814474, -3.0066595, 3.0066595)
1: (-6.8198452, -4.3332038, -6.8198452, -4.3332038, -2.4866414, 2.4866414)
2: (-8.8041763, -6.4693022, -8.8041763, -6.4693022, -2.3348742, 2.3348742)
3: (-10.1440105, -7.5077662, -10.1440105, -7.5077662, -2.6362443, 2.6362443)
4: (-5.0150108, -2.4794838, -5.0150108, -2.4794838, -2.5355270, 2.5355270)
5: (-5.4323273, -2.9420607, -5.4323273, -2.9420607, -2.4902666, 2.4902666)
6: (-13.7059364, -10.6496744, -13.7059364, -10.6496744, -3.0562620, 3.0562620)
7: (3.2413931, 5.0245962, 3.2413931, 5.0245962, -1.7832031, 1.7832031)
8: (-4.4893575, -1.5198970, -4.4893575, -1.5198970, -2.9694605, 2.9694605)
9: (-2.3653054, 0.1176131, -2.3653054, 0.1176131, -2.4829185, 2.4829185)

## BASE Result
execution time: IAR + LP analysis = 14.92 + 33.09 = 48.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.4649108, upper bound: 1.4649096


# Binary Search by BASE starts (time budget: 3551.98 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.7645337581634521
rel_dist={7: [-1.1515141861453313, 1.1515132652632292]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.6208463907241821
rel_dist={7: [-0.8562086879320736, 0.8562091799144329]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.6687421798706055
rel_dist={7: [-0.9703767544746134, 0.9703745834961706]}

## Binary Search Result
Binary search time: 143.98 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual) starts
Time budget: 3408.01 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576357, upper bound: 1.2635090
time: 5.11 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674282, upper bound: 1.2674287
time: 4.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.65
Output dim: 7, lower bound: -1.2576357, upper bound: 1.2635090
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.65
Output dim: 7, lower bound: -1.2674282, upper bound: 1.2674287

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.2558632, -6.2892208, -9.2799244, -6.2824721, -2.6974874, 2.7138042
1: -6.7975464, -4.3430395, -6.8150768, -4.3353853, -2.3956389, 2.4064968
2: -8.7745104, -6.4910307, -8.7966909, -6.4738235, -2.2791739, 2.2850246
3: -10.0992203, -7.5297594, -10.1332560, -7.5112233, -2.2687430, 2.2851782
4: -4.9898586, -2.5014129, -5.0091887, -2.4843678, -2.5054908, 2.5077758
5: -5.3845930, -2.9718044, -5.4209347, -2.9465370, -2.3943353, 2.4031699
6: -13.6843634, -10.6646891, -13.7013607, -10.6525679, -3.0317955, 3.0366716
7: 3.2800965, 5.0211525, 3.2509294, 5.0237904, -1.7436938, 1.7702231
8: -4.4638987, -1.5591264, -4.4852853, -1.5298858, -2.5648608, 2.5638356
9: -2.3446670, 0.0989108, -2.3608122, 0.1127104, -2.4573774, 2.4597230

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523109, upper bound: 1.2634973
time: 4.13 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576227, upper bound: 1.2634972
time: 4.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.2880926, -6.2814493, -9.2881069, -6.2814474, -2.7288570, 2.7337122
1: -6.8198409, -4.3332071, -6.8198452, -4.3332038, -2.4190784, 2.4236317
2: -8.8041601, -6.4693065, -8.8041763, -6.4693022, -2.3071690, 2.3116410
3: -10.1439934, -7.5077729, -10.1440105, -7.5077662, -2.3066638, 2.3186841
4: -5.0149994, -2.4794898, -5.0150108, -2.4794838, -2.5355155, 2.5355210
5: -5.4323173, -2.9420679, -5.4323273, -2.9420607, -2.4199920, 2.4447508
6: -13.7059288, -10.6496811, -13.7059364, -10.6496744, -3.0562544, 3.0562553
7: 3.2414083, 5.0245953, 3.2413931, 5.0245962, -1.7831879, 1.7832022
8: -4.4893527, -1.5199199, -4.4893575, -1.5198970, -2.6029711, 2.5897291
9: -2.3652968, 0.1176064, -2.3653054, 0.1176131, -2.4829099, 2.4829118

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621075, upper bound: 1.2674178
time: 4.49 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674151, upper bound: 1.2674162
time: 4.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.13 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -1.2523109, upper bound: 1.2634973
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -1.2576227, upper bound: 1.2634972
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -1.2621075, upper bound: 1.2674178
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 22.13
Output dim: 7, lower bound: -1.2674151, upper bound: 1.2674162

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2749214, -6.2827640, -2.6300821, 2.6959705
1: -6.7868700, -4.3616509, -6.8135500, -4.3376770, -2.3554001, 2.3611684
2: -8.7657557, -6.4971232, -8.7956333, -6.4747086, -2.2537255, 2.2710006
3: -10.0937233, -7.5504613, -10.1324072, -7.5138788, -2.2600265, 2.2614598
4: -4.9823031, -2.5226245, -5.0082731, -2.4870746, -2.4952285, 2.4856486
5: -5.3602338, -2.9799883, -5.4178495, -2.9475322, -2.3556623, 2.3757772
6: -13.6672230, -10.7211151, -13.6992979, -10.6595106, -3.0077124, 2.9781828
7: 3.2947273, 5.0180111, 3.2527618, 5.0234170, -1.7286897, 1.7652493
8: -4.4546123, -1.5902944, -4.4840426, -1.5337014, -2.5503578, 2.5310411
9: -2.3205733, 0.0936966, -2.3577976, 0.1119845, -2.4325578, 2.4514942

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523109, upper bound: 1.2581852
time: 4.14 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523109, upper bound: 1.2634973
time: 4.26 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396040, -9.2799110, -6.2824755, -2.7052445, 2.7465739
1: -6.8267632, -4.3343019, -6.8150730, -4.3353920, -2.4036965, 2.4491539
2: -8.7878790, -6.4858274, -8.7966881, -6.4738235, -2.2893381, 2.2968886
3: -10.1272221, -7.5233526, -10.1332541, -7.5112324, -2.2979207, 2.2904007
4: -5.0195088, -2.4906089, -5.0091858, -2.4843740, -2.5351348, 2.5185769
5: -5.3917122, -2.9466255, -5.4209309, -2.9465394, -2.4141235, 2.4147532
6: -13.7600527, -10.6568575, -13.7013607, -10.6525850, -3.1074677, 3.0445032
7: 3.2653699, 5.0516911, 3.2509356, 5.0237894, -1.7584195, 1.8007555
8: -4.5052948, -1.5559216, -4.4852839, -1.5298972, -2.6070809, 2.5666051
9: -2.3622165, 0.1295742, -2.3608041, 0.1127096, -2.4749260, 2.4903784

Time for backsubstitution: 13.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576227, upper bound: 1.2576201
time: 4.15 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576227, upper bound: 1.2634972
time: 4.25 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2831097, -6.2817378, -2.6618061, 2.7157807
1: -6.8073835, -4.3517179, -6.8183289, -4.3354921, -2.3770819, 2.3781779
2: -8.7954216, -6.4763594, -8.8031130, -6.4701843, -2.2817655, 2.2969728
3: -10.1369629, -7.5283790, -10.1431608, -7.5103340, -2.2962673, 2.2949109
4: -5.0075369, -2.5014136, -5.0140982, -2.4821897, -2.5253472, 2.5126846
5: -5.4072847, -2.9503319, -5.4292421, -2.9430604, -2.3807368, 2.4171681
6: -13.6888180, -10.7062511, -13.7038717, -10.6566191, -3.0321989, 2.9976206
7: 3.2560339, 5.0214624, 3.2432132, 5.0242224, -1.7681885, 1.7782493
8: -4.4790025, -1.5508962, -4.4881186, -1.5237064, -2.5872669, 2.5568831
9: -2.3410528, 0.1115867, -2.3622925, 0.1168859, -2.4579387, 2.4738793

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581846, upper bound: 1.2576202
time: 4.00 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581849, upper bound: 1.2576205
time: 4.08 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317038, -9.2880936, -6.2814474, -2.7374516, 2.7686107
1: -6.8491993, -4.3243461, -6.8198419, -4.3332100, -2.4269986, 2.4662690
2: -8.8176327, -6.4633989, -8.8041744, -6.4693046, -2.3172970, 2.3241067
3: -10.1720533, -7.5003195, -10.1440058, -7.5077753, -2.3358495, 2.3243020
4: -5.0447426, -2.4684753, -5.0150080, -2.4794893, -2.5652533, 2.5465326
5: -5.4397163, -2.9162662, -5.4323239, -2.9420631, -2.4401584, 2.4568784
6: -13.7818356, -10.6415138, -13.7059317, -10.6496916, -3.1321440, 3.0644178
7: 3.2260957, 5.0552692, 3.2413988, 5.0245962, -1.7985005, 1.8138704
8: -4.5310326, -1.5154562, -4.4893570, -1.5199089, -2.6440578, 2.5931861
9: -2.3830662, 0.1482708, -2.3652978, 0.1176099, -2.5006762, 2.5135686

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634963, upper bound: 1.2576201
time: 4.41 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634966, upper bound: 1.2650183
time: 4.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.37 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2523109, upper bound: 1.2581852
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2523109, upper bound: 1.2634973
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2576227, upper bound: 1.2576201
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2576227, upper bound: 1.2634972
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2581846, upper bound: 1.2576202
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2581849, upper bound: 1.2576205
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2634963, upper bound: 1.2576201
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.2634966, upper bound: 1.2650183

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2393131, -6.2848878, -2.6208782, 2.6372488
1: -6.7868700, -4.3616509, -6.8025484, -4.3539186, -2.3164830, 2.3254704
2: -8.7657557, -6.4971232, -8.7879572, -6.4808936, -2.2404985, 2.2470579
3: -10.0937233, -7.5504613, -10.1262169, -7.5320721, -2.2400029, 2.2547512
4: -4.9823031, -2.5226245, -5.0017071, -2.5063059, -2.4759972, 2.4790826
5: -5.3602338, -2.9799883, -5.3959103, -2.9547772, -2.3341031, 2.3425121
6: -13.6672230, -10.7211151, -13.6842489, -10.7091427, -2.9580803, 2.9631338
7: 3.2947273, 5.0180111, 3.2656455, 5.0206590, -1.7259316, 1.7523656
8: -4.4546123, -1.5902944, -4.4749084, -1.5609069, -2.5228748, 2.5202782
9: -2.3205733, 0.0936966, -2.3365693, 0.1066949, -2.4272683, 2.4302659

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2523218
time: 4.59 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2581826
time: 4.41 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2862339, -6.2327266, -2.6730998, 2.7018070
1: -6.7868700, -4.3616509, -6.8444753, -4.3265290, -2.3455105, 2.3734303
2: -8.7657557, -6.4971232, -8.8101902, -6.4678893, -2.2530885, 2.2827344
3: -10.0937233, -7.5504613, -10.1613188, -7.5040007, -2.2688460, 2.2915621
4: -4.9823031, -2.5226245, -5.0389423, -2.4733114, -2.5089917, 2.5163178
5: -5.3602338, -2.9799883, -5.4283252, -2.9207292, -2.3706007, 2.3754482
6: -13.6672230, -10.7211151, -13.7772856, -10.6443748, -3.0228481, 3.0561705
7: 3.2947273, 5.0180111, 3.2355833, 5.0544586, -1.7597313, 1.7824278
8: -4.4546123, -1.5902944, -4.5269942, -1.5254107, -2.5589747, 2.5755477
9: -2.3205733, 0.0936966, -2.3786323, 0.1433871, -2.4639604, 2.4723289

Time for backsubstitution: 12.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2576335
time: 4.78 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2634944
time: 4.73 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396040, -9.2558498, -6.2892222, -2.6963000, 2.7247498
1: -6.8267632, -4.3343019, -6.7975435, -4.3430452, -2.3967929, 2.4313922
2: -8.7878790, -6.4858274, -8.7745085, -6.4910326, -2.2727971, 2.2744942
3: -10.1272221, -7.5233526, -10.0992203, -7.5297689, -2.2788410, 2.2548826
4: -5.0195088, -2.4906089, -4.9898567, -2.5014193, -2.5180895, 2.4992478
5: -5.3917122, -2.9466255, -5.3845882, -2.9718065, -2.3863945, 2.3781886
6: -13.7600527, -10.6568575, -13.6843615, -10.6647043, -3.0953484, 3.0275040
7: 3.2653699, 5.0516911, 3.2801032, 5.0211515, -1.7557817, 1.7715878
8: -4.5052948, -1.5559216, -4.4638963, -1.5591373, -2.5804448, 2.5404170
9: -2.3622165, 0.1295742, -2.3446574, 0.0989106, -2.4611270, 2.4742317

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576157
time: 4.12 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576182, upper bound: 1.2576159
time: 4.39 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396040, -9.2880783, -6.2814512, -2.7074275, 2.7519383
1: -6.8267632, -4.3343019, -6.8198366, -4.3332138, -2.4065557, 2.4519680
2: -8.7878790, -6.4858274, -8.8041563, -6.4693079, -2.2919850, 2.3043001
3: -10.1272221, -7.5233526, -10.1439924, -7.5077801, -2.3011584, 2.3015764
4: -5.0195088, -2.4906089, -5.0149965, -2.4794970, -2.5400119, 2.5243876
5: -5.3917122, -2.9466255, -5.4323149, -2.9420695, -2.4162235, 2.4264967
6: -13.7600527, -10.6568575, -13.7059269, -10.6496964, -3.1103563, 3.0490694
7: 3.2653699, 5.0516911, 3.2414131, 5.0245943, -1.7592244, 1.8102779
8: -4.5052948, -1.5559216, -4.4893498, -1.5199294, -2.6106400, 2.5691683
9: -2.3622165, 0.1295742, -2.3652871, 0.1176050, -2.4798214, 2.4948611

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517570, upper bound: 1.2634929
time: 4.47 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576182, upper bound: 1.2634930
time: 4.26 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2508535, -6.2895060, -2.6552176, 2.6817281
1: -6.8073835, -4.3517179, -6.7959852, -4.3453388, -2.3673749, 2.3532164
2: -8.7954216, -6.4763594, -8.7734518, -6.4919066, -2.2670584, 2.2671552
3: -10.1369629, -7.5283790, -10.0983686, -7.5324035, -2.2859697, 2.2481937
4: -5.0075369, -2.5014136, -4.9889369, -2.5041223, -2.5034146, 2.4875233
5: -5.4072847, -2.9503319, -5.3815117, -2.9727960, -2.3756518, 2.3687844
6: -13.6888180, -10.7062511, -13.6822901, -10.6716309, -3.0171871, 2.9760389
7: 3.2560339, 5.0214624, 3.2819610, 5.0207777, -1.7647438, 1.7395015
8: -4.4790025, -1.5508962, -4.4626369, -1.5629597, -2.5506406, 2.5416546
9: -2.3410528, 0.1115867, -2.3416629, 0.0981911, -2.4392438, 2.4532495

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581846, upper bound: 1.2523085
time: 4.29 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581846, upper bound: 1.2576202
time: 4.08 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2830954, -6.2817407, -2.6618013, 2.7109778
1: -6.8073835, -4.3517179, -6.8183250, -4.3354955, -2.3812437, 2.3781700
2: -8.7954216, -6.4763594, -8.8030968, -6.4701891, -2.2817602, 2.2925804
3: -10.1369629, -7.5283790, -10.1431456, -7.5103388, -2.2962623, 2.2829340
4: -5.0075369, -2.5014136, -5.0140867, -2.4821954, -2.5253415, 2.5126731
5: -5.4072847, -2.9503319, -5.4292321, -2.9430668, -2.3807254, 2.3926446
6: -13.6888180, -10.7062511, -13.7038651, -10.6566229, -3.0321951, 2.9976139
7: 3.2560339, 5.0214624, 3.2432284, 5.0242219, -1.7681880, 1.7782340
8: -4.4790025, -1.5508962, -4.4881134, -1.5237279, -2.5737720, 2.5568745
9: -2.3410528, 0.1115867, -2.3622816, 0.1168799, -2.4579327, 2.4738684

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581849, upper bound: 1.2597152
time: 4.40 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581849, upper bound: 1.2576205
time: 4.16 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317038, -9.2558498, -6.2892222, -2.7311635, 2.7345912
1: -6.8491993, -4.3243461, -6.7975435, -4.3430452, -2.4172316, 2.4413400
2: -8.8176327, -6.4633989, -8.7745085, -6.4910326, -2.3025665, 2.2942858
3: -10.1720533, -7.5003195, -10.0992203, -7.5297689, -2.3255424, 2.2775950
4: -5.0447426, -2.4684753, -4.9898567, -2.5014193, -2.5433233, 2.5213814
5: -5.4397163, -2.9162662, -5.3845882, -2.9718065, -2.4350753, 2.4085617
6: -13.7818356, -10.6415138, -13.6843615, -10.6647043, -3.1171312, 3.0428476
7: 3.2260957, 5.0552692, 3.2801032, 5.0211515, -1.7950559, 1.7751660
8: -4.5310326, -1.5154562, -4.4638963, -1.5591373, -2.6062942, 2.5778170
9: -2.3830662, 0.1482708, -2.3446574, 0.0989106, -2.4819767, 2.4929283

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2576158
time: 4.49 seconds

## Relational analysis of IS_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634918, upper bound: 1.2576160
time: 4.64 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317038, -9.2880783, -6.2814512, -2.7374477, 2.7659302
1: -6.8491993, -4.3243461, -6.8198366, -4.3332138, -2.4315052, 2.4662616
2: -8.8176327, -6.4633989, -8.8041563, -6.4693079, -2.3172903, 2.3197432
3: -10.1720533, -7.5003195, -10.1439924, -7.5077801, -2.3358440, 2.3122742
4: -5.0447426, -2.4684753, -5.0149965, -2.4794970, -2.5652456, 2.5465212
5: -5.4397163, -2.9162662, -5.4323149, -2.9420695, -2.4401469, 2.4322758
6: -13.7818356, -10.6415138, -13.7059269, -10.6496964, -3.1321392, 3.0644131
7: 3.2260957, 5.0552692, 3.2414131, 5.0245943, -1.7984986, 1.8138561
8: -4.5310326, -1.5154562, -4.4893498, -1.5199294, -2.6331162, 2.5931768
9: -2.3830662, 0.1482708, -2.3652871, 0.1176050, -2.5006711, 2.5135579

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2650144
time: 4.59 seconds

## Relational analysis of IS_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634921, upper bound: 1.2650142
time: 4.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.38 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2523218
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2581826
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2576335
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2634944
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576157
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2576182, upper bound: 1.2576159
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2517570, upper bound: 1.2634929
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2576182, upper bound: 1.2634930
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2581846, upper bound: 1.2523085
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2581846, upper bound: 1.2576202
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2581849, upper bound: 1.2597152
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2581849, upper bound: 1.2576205
IS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2576158
IS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2634918, upper bound: 1.2576160
IS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2650144
IS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.38
Output dim: 7, lower bound: -1.2634921, upper bound: 1.2650142

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2378464, -6.3026524, -2.6031055, 2.6353087
1: -6.7868700, -4.3616509, -6.8010497, -4.3550100, -2.3154311, 2.3239541
2: -8.7657557, -6.4971232, -8.7863140, -6.4913125, -2.2261610, 2.2398036
3: -10.0937233, -7.5504613, -10.1252527, -7.5481896, -2.2225542, 2.2498815
4: -4.9823031, -2.5226245, -4.9961500, -2.5085464, -2.4737568, 2.4735255
5: -5.3602338, -2.9799883, -5.3921118, -2.9767818, -2.3121138, 2.3383629
6: -13.6672230, -10.7211151, -13.6687059, -10.7102346, -2.9569883, 2.9475908
7: 3.2947273, 5.0180111, 3.2726197, 5.0193686, -1.7246413, 1.7453914
8: -4.4546123, -1.5902944, -4.4711232, -1.5709162, -2.5096769, 2.5128298
9: -2.3205733, 0.0936966, -2.3270938, 0.1059012, -2.4264746, 2.4207904

Time for backsubstitution: 14.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2464447
time: 4.48 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2523217
time: 4.91 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916131, -9.2826748, -6.2830811, -2.6251130, 2.6799459
1: -6.7868690, -4.3616514, -6.8113551, -4.3509269, -2.3196254, 2.3346205
2: -8.7657547, -6.4971280, -8.8114977, -6.4764667, -2.2573714, 2.2644997
3: -10.0937204, -7.5504661, -10.1622047, -7.5292969, -2.2498968, 2.2873888
4: -4.9823008, -2.5226254, -5.0040612, -2.4965551, -2.4857457, 2.4814358
5: -5.3602324, -2.9799948, -5.4529486, -2.9537749, -2.3378744, 2.3842123
6: -13.6672192, -10.7211161, -13.6901283, -10.6690788, -2.9981403, 2.9690123
7: 3.2947292, 5.0180106, 3.2631536, 5.0358996, -1.7411704, 1.7548571
8: -4.4546103, -1.5902982, -4.5025253, -1.5578218, -2.5365314, 2.5443342
9: -2.3205700, 0.0936956, -2.3402143, 0.1255753, -2.4461453, 2.4339099

Time for backsubstitution: 14.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2523056
time: 4.36 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2581826
time: 4.40 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2847652, -6.2505021, -2.6552987, 2.6998575
1: -6.7868700, -4.3616509, -6.8429656, -4.3275976, -2.3444910, 2.3718884
2: -8.7657557, -6.4971232, -8.8085737, -6.4782786, -2.2387609, 2.2755251
3: -10.0937233, -7.5504613, -10.1603298, -7.5201378, -2.2514009, 2.2866933
4: -4.9823031, -2.5226245, -5.0334201, -2.4755089, -2.5067942, 2.5107956
5: -5.3602338, -2.9799883, -5.4245520, -2.9427199, -2.3486247, 2.3712997
6: -13.6672230, -10.7211151, -13.7616920, -10.6454554, -3.0217676, 3.0405769
7: 3.2947273, 5.0180111, 3.2425385, 5.0531502, -1.7584229, 1.7754726
8: -4.4546123, -1.5902944, -4.5231619, -1.5354137, -2.5457706, 2.5679941
9: -2.3205733, 0.0936966, -2.3691497, 0.1425672, -2.4631405, 2.4628463

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2517560
time: 4.54 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2576332
time: 4.45 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916131, -9.3296461, -6.2309604, -2.6773562, 2.7366772
1: -6.7868690, -4.3616514, -6.8532681, -4.3234448, -2.3487897, 2.3825653
2: -8.7657547, -6.4971280, -8.8337879, -6.4633064, -2.2701211, 2.3003054
3: -10.0937204, -7.5504661, -10.1972008, -7.5012407, -2.2786963, 2.3139036
4: -4.9823008, -2.5226254, -5.0413690, -2.4634609, -2.5188398, 2.5187435
5: -5.3602324, -2.9799948, -5.4854393, -2.9197016, -2.3743997, 2.4140823
6: -13.6672192, -10.7211161, -13.7828550, -10.6042137, -3.0630054, 3.0617390
7: 3.2947292, 5.0180106, 3.2330408, 5.0696354, -1.7749062, 1.7849698
8: -4.4546103, -1.5902982, -4.5544519, -1.5222950, -2.5726571, 2.5814672
9: -2.3205700, 0.0936956, -2.3823972, 0.1622107, -2.4827807, 2.4760928

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2576171
time: 4.45 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2634941
time: 4.50 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.2558498, -6.2892222, -2.6943493, 2.7069590
1: -6.8252726, -4.3353658, -6.7975435, -4.3430452, -2.3952904, 2.4303145
2: -8.7862673, -6.4962525, -8.7745085, -6.4910326, -2.2655997, 2.2601616
3: -10.1262522, -7.5394750, -10.0992203, -7.5297689, -2.2739525, 2.2374332
4: -5.0140319, -2.4928031, -4.9898567, -2.5014193, -2.5126126, 2.4970536
5: -5.3879519, -2.9686193, -5.3845882, -2.9718065, -2.3823452, 2.3562117
6: -13.7444639, -10.6579409, -13.6843615, -10.6647043, -3.0797596, 3.0264206
7: 3.2723179, 5.0503802, 3.2801032, 5.0211515, -1.7488337, 1.7702770
8: -4.5014772, -1.5659080, -4.4638963, -1.5591373, -2.5729475, 2.5272262
9: -2.3527238, 0.1287827, -2.3446574, 0.0989106, -2.4516344, 2.4734402

Time for backsubstitution: 14.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2517575
time: 4.32 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576188
time: 4.45 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.3049059, -6.2378545, -9.2558498, -6.2892294, -2.7317176, 2.7289860
1: -6.8355908, -4.3312049, -6.7975411, -4.3430471, -2.4058385, 2.4344091
2: -8.8114977, -6.4812555, -8.7745075, -6.4910359, -2.2903976, 2.2913306
3: -10.1630249, -7.5206203, -10.0992184, -7.5297751, -2.3069611, 2.2647567
4: -5.0219293, -2.4807301, -4.9898539, -2.5014200, -2.5205092, 2.5091238
5: -5.4488506, -2.9456134, -5.3845882, -2.9718146, -2.4230695, 2.3819780
6: -13.7656345, -10.6166916, -13.6843567, -10.6647034, -3.1009312, 3.0676651
7: 3.2628231, 5.0668783, 3.2801046, 5.0211515, -1.7583284, 1.7867737
8: -4.5327868, -1.5527935, -4.4638948, -1.5591416, -2.5861392, 2.5541337
9: -2.3659363, 0.1483328, -2.3446560, 0.0989097, -2.4648461, 2.4929888

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576181, upper bound: 1.2523070
time: 4.20 seconds

## Relational analysis of IS_A1_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576184, upper bound: 1.2523070
time: 4.45 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.2880783, -6.2814512, -2.7054768, 2.7341473
1: -6.8252726, -4.3353658, -6.8198366, -4.3332138, -2.4050531, 2.4508903
2: -8.7862673, -6.4962525, -8.8041563, -6.4693079, -2.2847877, 2.2899675
3: -10.1262522, -7.5394750, -10.1439924, -7.5077801, -2.2962699, 2.2841270
4: -5.0140319, -2.4928031, -5.0149965, -2.4794970, -2.5345349, 2.5221934
5: -5.3879519, -2.9686193, -5.4323149, -2.9420695, -2.4121742, 2.4045203
6: -13.7444639, -10.6579409, -13.7059269, -10.6496964, -3.0947676, 3.0479860
7: 3.2723179, 5.0503802, 3.2414131, 5.0245943, -1.7522764, 1.8089671
8: -4.5014772, -1.5659080, -4.4893498, -1.5199294, -2.6033382, 2.5559773
9: -2.3527238, 0.1287827, -2.3652871, 0.1176050, -2.4703288, 2.4940698

Time for backsubstitution: 14.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576315
time: 4.52 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517570, upper bound: 1.2634929
time: 4.42 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.3049059, -6.2378545, -9.2880783, -6.2814589, -2.7410169, 2.7561746
1: -6.8355908, -4.3312049, -6.8198376, -4.3332143, -2.4156008, 2.4549863
2: -8.8114977, -6.4812555, -8.8041563, -6.4693117, -2.3095860, 2.3211348
3: -10.1630249, -7.5206203, -10.1439896, -7.5077877, -2.3243084, 2.3114507
4: -5.0219293, -2.4807301, -5.0149946, -2.4794972, -2.5424321, 2.5342646
5: -5.4488506, -2.9456134, -5.4323139, -2.9420757, -2.4415679, 2.4302859
6: -13.7656345, -10.6166916, -13.7059231, -10.6496973, -3.1159372, 3.0892315
7: 3.2628231, 5.0668783, 3.2414150, 5.0245943, -1.7617712, 1.8254633
8: -4.5327868, -1.5527935, -4.4893489, -1.5199347, -2.6116815, 2.5828841
9: -2.3659363, 0.1483328, -2.3652844, 0.1176052, -2.4835415, 2.5136173

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576181, upper bound: 1.2581810
time: 4.11 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576184, upper bound: 1.2581810
time: 4.27 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2152367, -6.2916050, -2.6460896, 2.6231344
1: -6.8073835, -4.3517179, -6.7868700, -4.3616509, -2.3284178, 2.3193645
2: -8.7954216, -6.4763594, -8.7657557, -6.4971232, -2.2544866, 2.2431145
3: -10.1369629, -7.5283790, -10.0937233, -7.5504613, -2.2659681, 2.2432177
4: -5.0075369, -2.5014136, -4.9823031, -2.5226245, -2.4849124, 2.4808896
5: -5.4072847, -2.9503319, -5.3602338, -2.9799883, -2.3542581, 2.3361065
6: -13.6888180, -10.7062511, -13.6672230, -10.7211151, -2.9677029, 2.9609718
7: 3.2560339, 5.0214624, 3.2947273, 5.0180111, -1.7619772, 1.7267351
8: -4.4790025, -1.5508962, -4.4546123, -1.5902944, -2.5229402, 2.5322011
9: -2.3410528, 0.1115867, -2.3205733, 0.0936966, -2.4347494, 2.4321599

Time for backsubstitution: 14.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523191, upper bound: 1.2523064
time: 5.77 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581800, upper bound: 1.2523074
time: 4.25 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2614956, -6.2396040, -2.6907561, 2.6867905
1: -6.8073835, -4.3517179, -6.8267632, -4.3343019, -2.3574038, 2.3655035
2: -8.7954216, -6.4763594, -8.7878790, -6.4858274, -2.2656183, 2.2788997
3: -10.1369629, -7.5283790, -10.1272221, -7.5233526, -2.2943616, 2.2782960
4: -5.0075369, -2.5014136, -5.0195088, -2.4906089, -2.5169280, 2.5180953
5: -5.4072847, -2.9503319, -5.3917122, -2.9466255, -2.3899803, 2.3679535
6: -13.6888180, -10.7062511, -13.7600527, -10.6568575, -3.0319605, 3.0538015
7: 3.2560339, 5.0214624, 3.2653699, 5.0516911, -1.7956572, 1.7560925
8: -4.4790025, -1.5508962, -4.5052948, -1.5559216, -2.5584450, 2.5793719
9: -2.3410528, 0.1115867, -2.3622165, 0.1295742, -2.4706268, 2.4738030

Time for backsubstitution: 14.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523191, upper bound: 1.2576193
time: 4.13 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2581800, upper bound: 1.2576191
time: 4.29 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2475052, -6.2838678, -2.6526451, 2.6526451
1: -6.8073835, -4.3517179, -6.8073835, -4.3517179, -2.3422647, 2.3422649
2: -8.7954216, -6.4763594, -8.7954216, -6.4763594, -2.2685890, 2.2685893
3: -10.1369629, -7.5283790, -10.1369629, -7.5283790, -2.2762637, 2.2762640
4: -5.0075369, -2.5014136, -5.0075369, -2.5014136, -2.5061233, 2.5061233
5: -5.4072847, -2.9503319, -5.4072847, -2.9503319, -2.3593602, 2.3593600
6: -13.6888180, -10.7062511, -13.6888180, -10.7062511, -2.9825668, 2.9825668
7: 3.2560339, 5.0214624, 3.2560339, 5.0214624, -1.7654285, 1.7654285
8: -4.4790025, -1.5508962, -4.4790025, -1.5508962, -2.5460153, 2.5460150
9: -2.3410528, 0.1115867, -2.3410528, 0.1115867, -2.4526396, 2.4526396

Time for backsubstitution: 14.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621035, upper bound: 1.2538699
time: 4.88 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2597089
time: 4.95 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2944183, -6.2317038, -2.7047176, 2.7168102
1: -6.8073835, -4.3517179, -6.8491993, -4.3243461, -2.3711486, 2.3903441
2: -8.7954216, -6.4763594, -8.8176327, -6.4633989, -2.2811780, 2.3042901
3: -10.1369629, -7.5283790, -10.1720533, -7.5003195, -2.3050606, 2.3130319
4: -5.0075369, -2.5014136, -5.0447426, -2.4684753, -2.5390615, 2.5433290
5: -5.4072847, -2.9503319, -5.4397163, -2.9162662, -2.3957601, 2.3923354
6: -13.6888180, -10.7062511, -13.7818356, -10.6415138, -3.0473042, 3.0755844
7: 3.2560339, 5.0214624, 3.2260957, 5.0552692, -1.7992353, 1.7953668
8: -4.4790025, -1.5508962, -4.5310326, -1.5154562, -2.5822411, 2.6014810
9: -2.3410528, 0.1115867, -2.3830662, 0.1482708, -2.4893236, 2.4946527

Time for backsubstitution: 14.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621032, upper bound: 1.2591742
time: 4.55 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2650134
time: 5.63 seconds

## BFS IS instance: IS_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.2558498, -6.2892222, -2.7292051, 2.7167830
1: -6.8476830, -4.3254142, -6.7975435, -4.3430452, -2.4156857, 2.4402614
2: -8.8160028, -6.4737835, -8.7745085, -6.4910326, -2.2953482, 2.2799680
3: -10.1710596, -7.5164576, -10.0992203, -7.5297689, -2.3206797, 2.2601480
4: -5.0392056, -2.4706733, -4.9898567, -2.5014193, -2.5377862, 2.5191834
5: -5.4359431, -2.9382565, -5.3845882, -2.9718065, -2.4309702, 2.3865886
6: -13.7662449, -10.6425924, -13.6843615, -10.6647043, -3.1015406, 3.0417690
7: 3.2330532, 5.0539637, 3.2801032, 5.0211515, -1.7880983, 1.7738605
8: -4.5272012, -1.5254626, -4.4638963, -1.5591373, -2.5989170, 2.5646131
9: -2.3735833, 0.1474495, -2.3446574, 0.0989106, -2.4724939, 2.4921069

Time for backsubstitution: 14.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2517576
time: 4.49 seconds

## Relational analysis of IS_A2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576307, upper bound: 1.2576190
time: 4.52 seconds

## BFS IS instance: IS_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.3378305, -6.2299337, -9.2558498, -6.2892294, -2.7596059, 2.7388422
1: -6.8579874, -4.3212638, -6.7975411, -4.3430471, -2.4263296, 2.4443598
2: -8.8412151, -6.4588223, -8.7745075, -6.4910359, -2.3201375, 2.3111060
3: -10.2079506, -7.4975433, -10.0992184, -7.5297751, -2.3423104, 2.2874305
4: -5.0471773, -2.4586329, -4.9898539, -2.5014200, -2.5457573, 2.5312209
5: -5.4968190, -2.9152360, -5.3845882, -2.9718146, -2.4529591, 2.4123740
6: -13.7874041, -10.6013517, -13.6843567, -10.6647034, -3.1227007, 3.0830050
7: 3.2235560, 5.0704441, 3.2801046, 5.0211515, -1.7975955, 1.7903395
8: -4.5584836, -1.5123401, -4.4638948, -1.5591416, -2.6073465, 2.5914817
9: -2.3868451, 0.1671113, -2.3446560, 0.0989097, -2.4857550, 2.5117674

Time for backsubstitution: 14.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634918, upper bound: 1.2523072
time: 4.66 seconds

## Relational analysis of IS_A2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634920, upper bound: 1.2523072
time: 4.68 seconds

## BFS IS instance: IS_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.2880783, -6.2814512, -2.7354898, 2.7481220
1: -6.8476830, -4.3254142, -6.8198366, -4.3332138, -2.4299545, 2.4651918
2: -8.8160028, -6.4737835, -8.8041563, -6.4693079, -2.3100715, 2.3053839
3: -10.1710596, -7.5164576, -10.1439924, -7.5077801, -2.3309827, 2.2948272
4: -5.0392056, -2.4706733, -5.0149965, -2.4794970, -2.5597086, 2.5443232
5: -5.4359431, -2.9382565, -5.4323149, -2.9420695, -2.4360414, 2.4102945
6: -13.7662449, -10.6425924, -13.7059269, -10.6496964, -3.1165485, 3.0633345
7: 3.2330532, 5.0539637, 3.2414131, 5.0245943, -1.7915411, 1.8125505
8: -4.5272012, -1.5254626, -4.4893498, -1.5199294, -2.6255460, 2.5799956
9: -2.3735833, 0.1474495, -2.3652871, 0.1176050, -2.4911883, 2.5127366

Time for backsubstitution: 14.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2615630, upper bound: 1.2591761
time: 4.62 seconds

## Relational analysis of IS_A2_A2_B2_A1_B2

### Relational analysis result of IS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2615628, upper bound: 1.2650156
time: 4.77 seconds

## BFS IS instance: IS_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.3378305, -6.2299337, -9.2880783, -6.2814589, -2.7731242, 2.7701817
1: -6.8579874, -4.3212638, -6.8198376, -4.3332143, -2.4407816, 2.4692969
2: -8.8412151, -6.4588223, -8.8041563, -6.4693117, -2.3348613, 2.3365638
3: -10.2079506, -7.4975433, -10.1439896, -7.5077877, -2.3653779, 2.3221111
4: -5.0471773, -2.4586329, -5.0149946, -2.4794972, -2.5676801, 2.5563617
5: -5.4968190, -2.9152360, -5.4323139, -2.9420757, -2.4779844, 2.4360673
6: -13.7874041, -10.6013517, -13.7059231, -10.6496973, -3.1377068, 3.1045713
7: 3.2235560, 5.0704441, 3.2414150, 5.0245943, -1.8010383, 1.8290291
8: -4.5584836, -1.5123401, -4.4893489, -1.5199347, -2.6394472, 2.6068833
9: -2.3868451, 0.1671113, -2.3652844, 0.1176052, -2.5044503, 2.5323958

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674107, upper bound: 1.2597091
time: 4.72 seconds

## Relational analysis of IS_A2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674109, upper bound: 1.2597110
time: 4.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.03 seconds
IS_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2464447
IS_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2523217
IS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2523056
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2581826
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2517560
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2576332
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2576171
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2634941
IS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2517575
IS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576188
IS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2576181, upper bound: 1.2523070
IS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2576184, upper bound: 1.2523070
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576315
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2517570, upper bound: 1.2634929
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2576181, upper bound: 1.2581810
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2576184, upper bound: 1.2581810
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523191, upper bound: 1.2523064
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2581800, upper bound: 1.2523074
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2523191, upper bound: 1.2576193
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2581800, upper bound: 1.2576191
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2621035, upper bound: 1.2538699
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2597089
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2621032, upper bound: 1.2591742
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2650134
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2517576
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2576307, upper bound: 1.2576190
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2634918, upper bound: 1.2523072
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2634920, upper bound: 1.2523072
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2615630, upper bound: 1.2591761
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2615628, upper bound: 1.2650156
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2674107, upper bound: 1.2597091
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.03
Output dim: 7, lower bound: -1.2674109, upper bound: 1.2597110

## BFS IS instance: IS_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2137794, -6.3093643, -2.5940318, 2.6098568
1: -6.7868700, -4.3616509, -6.7853842, -4.3627400, -2.3085446, 2.3080688
2: -8.7657557, -6.4971232, -8.7641144, -6.5075798, -2.2102990, 2.2173998
3: -10.0937233, -7.5504613, -10.0927734, -7.5665693, -2.2035093, 2.2160721
4: -4.9823031, -2.5226245, -4.9767933, -2.5248652, -2.4574380, 2.4541688
5: -5.3602338, -2.9799883, -5.3564520, -3.0019968, -2.2844992, 2.3023586
6: -13.6672230, -10.7211151, -13.6516829, -10.7222109, -2.9450121, 2.9305677
7: 3.2947273, 5.0180111, 3.3016930, 5.0167184, -1.7219911, 1.7163181
8: -4.4546123, -1.5902944, -4.4508424, -1.6002913, -2.4822021, 2.4879520
9: -2.3205733, 0.0936966, -2.3110857, 0.0929170, -2.4134903, 2.4047823

Time for backsubstitution: 14.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2464419
time: 5.18 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2464429
time: 4.21 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2460384, -6.3016338, -2.6053586, 2.6441474
1: -6.7868700, -4.3616509, -6.8058748, -4.3528070, -2.3183146, 2.3268828
2: -8.7657557, -6.4971232, -8.7937784, -6.4867759, -2.2287922, 2.2472281
3: -10.0937233, -7.5504613, -10.1359940, -7.5445051, -2.2257752, 2.2611043
4: -4.9823031, -2.5226245, -5.0019689, -2.5036554, -2.4786477, 2.4793444
5: -5.3602338, -2.9799883, -5.4034872, -2.9723353, -2.3141208, 2.3500755
6: -13.6672230, -10.7211151, -13.6732759, -10.7073421, -2.9598808, 2.9521608
7: 3.2947273, 5.0180111, 3.2630086, 5.0201755, -1.7254481, 1.7550025
8: -4.4546123, -1.5902944, -4.4752202, -1.5609097, -2.5190058, 2.5154781
9: -2.3205733, 0.0936966, -2.3315768, 0.1107858, -2.4313593, 2.4252734

Time for backsubstitution: 14.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2523191
time: 5.22 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2523218
time: 4.22 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916131, -9.2585421, -6.2898102, -2.6160188, 2.6544001
1: -6.7868690, -4.3616514, -6.7935686, -4.3586483, -2.3127542, 2.3167586
2: -8.7657547, -6.4971280, -8.7893171, -6.4936047, -2.2408390, 2.2421489
3: -10.0937204, -7.5504661, -10.1280594, -7.5477257, -2.2308807, 2.2517414
4: -4.9823008, -2.5226254, -4.9846506, -2.5135965, -2.4687042, 2.4620252
5: -5.3602324, -2.9799948, -5.4166589, -2.9790025, -2.3102508, 2.3574414
6: -13.6672192, -10.7211161, -13.6731148, -10.6811857, -2.9860334, 2.9519987
7: 3.2947292, 5.0180106, 3.2925806, 5.0332623, -1.7385330, 1.7254300
8: -4.4546103, -1.5902982, -4.4810233, -1.5871987, -2.5090771, 2.5180564
9: -2.3205700, 0.0936956, -2.3241744, 0.1117505, -2.4323206, 2.4178700

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2523031
time: 5.38 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2470544
time: 4.48 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 24.39 seconds
IS_A1_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 24.39
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2464419
IS_A1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 24.39
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2464429
IS_A1_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 24.39
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2523191
IS_A1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 24.39
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2523218
IS_A1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 24.39
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2523031
IS_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 24.39
Output dim: 7, lower bound: -1.2464457, upper bound: 1.2470544
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2523070, upper bound: 1.2581826
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2517560
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2576332
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2576171
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2634941
IS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2517575
IS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576188
IS_A1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2576181, upper bound: 1.2523070
IS_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2576184, upper bound: 1.2523070
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2576315
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2517570, upper bound: 1.2634929
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2576181, upper bound: 1.2581810
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2576184, upper bound: 1.2581810
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2523191, upper bound: 1.2523064
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2581800, upper bound: 1.2523074
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2523191, upper bound: 1.2576193
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2581800, upper bound: 1.2576191
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2621035, upper bound: 1.2538699
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2597089
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2621032, upper bound: 1.2591742
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2650134
IS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2517576
IS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2576307, upper bound: 1.2576190
IS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2634918, upper bound: 1.2523072
IS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2634920, upper bound: 1.2523072
IS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2615630, upper bound: 1.2591761
IS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2615628, upper bound: 1.2650156
IS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2674107, upper bound: 1.2597091
IS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 24.39
Output dim: 7, lower bound: -1.2674109, upper bound: 1.2597110
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.783203125
rel_dist={7: [-1.2674345485483558, 1.2674366290257746]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642620, upper bound: 1.0684015
time: 4.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712136, upper bound: 1.0712131
time: 4.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.83
Output dim: 7, lower bound: -1.0642620, upper bound: 1.0684015
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.83
Output dim: 7, lower bound: -1.0712136, upper bound: 1.0712131

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.2558632, -6.2892208, -9.2745152, -6.2831726, -2.3882356, 2.4002461
1: -6.7975464, -4.3430395, -6.8118286, -4.3368421, -2.1800194, 2.1888895
2: -8.7745104, -6.4910307, -8.7917318, -6.4769073, -2.0686626, 2.0722599
3: -10.0992203, -7.5297594, -10.1260633, -7.5135274, -2.0077040, 2.0187922
4: -4.9898586, -2.5014129, -5.0052977, -2.4877074, -2.4048934, 2.4020104
5: -5.3845930, -2.9718044, -5.4132886, -2.9496050, -2.1549468, 2.1590798
6: -13.6843634, -10.6646891, -13.6983089, -10.6545372, -3.0298262, 3.0328574
7: 3.2800965, 5.0211525, 3.2572799, 5.0232515, -1.6753802, 1.6963565
8: -4.4638987, -1.5591264, -4.4825168, -1.5365381, -2.2811613, 2.2829957
9: -2.3446670, 0.0989108, -2.3577750, 0.1094579, -2.3754382, 2.3768625

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614196, upper bound: 1.0683957
time: 5.24 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642552, upper bound: 1.0683947
time: 4.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.2880926, -6.2814493, -9.2881050, -6.2814484, -2.4201927, 2.4259524
1: -6.8198409, -4.3332071, -6.8198438, -4.3332052, -2.2045274, 2.2082031
2: -8.8041601, -6.4693065, -8.8041716, -6.4693031, -2.0977020, 2.1038010
3: -10.1439934, -7.5077729, -10.1440039, -7.5077682, -2.0455220, 2.0597854
4: -5.0149994, -2.4794898, -5.0150080, -2.4794850, -2.4349585, 2.4363351
5: -5.4323173, -2.9420679, -5.4323244, -2.9420626, -2.1775274, 2.2084942
6: -13.7059288, -10.6496811, -13.7059326, -10.6496773, -3.0562515, 3.0562515
7: 3.2414083, 5.0245953, 3.2413960, 5.0245972, -1.7166255, 1.7169366
8: -4.4893527, -1.5199199, -4.4893560, -1.5199018, -2.3255625, 2.3079486
9: -2.3652968, 0.1176064, -2.3653035, 0.1176124, -2.4056935, 2.4040990

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682620, upper bound: 1.0712061
time: 5.27 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712065, upper bound: 1.0712057
time: 4.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.33 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 22.33
Output dim: 7, lower bound: -1.0614196, upper bound: 1.0683957
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 22.33
Output dim: 7, lower bound: -1.0642552, upper bound: 1.0683947
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 22.33
Output dim: 7, lower bound: -1.0682620, upper bound: 1.0712061
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 22.33
Output dim: 7, lower bound: -1.0712065, upper bound: 1.0712057

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2595949, -6.2840395, -2.3199420, 2.3710189
1: -6.7868700, -4.3616509, -6.8072443, -4.3436666, -2.1298275, 2.1351764
2: -8.7657557, -6.4971232, -8.7885580, -6.4795399, -2.0407047, 2.0551448
3: -10.0937233, -7.5504613, -10.1235170, -7.5214386, -1.9935894, 1.9932506
4: -4.9823031, -2.5226245, -5.0025597, -2.4957762, -2.3714314, 2.3580799
5: -5.3602338, -2.9799883, -5.4041004, -2.9526019, -2.1108518, 2.1238904
6: -13.6672230, -10.7211151, -13.6921024, -10.6752472, -2.9709263, 2.9709873
7: 3.2947273, 5.0180111, 3.2627497, 5.0221281, -1.6658802, 1.6885239
8: -4.4546123, -1.5902944, -4.4787745, -1.5479269, -2.2594099, 2.2478068
9: -2.3205733, 0.0936966, -2.3488007, 0.1072819, -2.3470578, 2.3621325

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
time: 10.25 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
time: 6.51 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396040, -9.2744942, -6.2831717, -2.3835201, 2.4280930
1: -6.8267632, -4.3343019, -6.8118215, -4.3368540, -2.1880646, 2.2208152
2: -8.7878790, -6.4858274, -8.7917299, -6.4769111, -2.0788240, 2.0822008
3: -10.1272221, -7.5233526, -10.1260605, -7.5135422, -2.0368769, 2.0234559
4: -5.0195088, -2.4906089, -5.0052934, -2.4877179, -2.4219341, 2.4256420
5: -5.3917122, -2.9466255, -5.4132824, -2.9496078, -2.1660275, 2.1706538
6: -13.7600527, -10.6568575, -13.6983042, -10.6545620, -3.1054907, 3.0362830
7: 3.2653699, 5.0516911, 3.2572918, 5.0232506, -1.6893973, 1.7279829
8: -4.5052948, -1.5559216, -4.4825120, -1.5365582, -2.3184929, 2.2840562
9: -2.3622165, 0.1295742, -2.3577621, 0.1094546, -2.3905144, 2.4075780

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0683158
time: 4.76 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642525, upper bound: 1.0683926
time: 4.43 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2731953, -6.2823195, -2.3523107, 2.3966331
1: -6.8073835, -4.3517179, -6.8153038, -4.3400154, -2.1525769, 2.1542933
2: -8.7954216, -6.4763594, -8.8009872, -6.4719243, -2.0697975, 2.0860400
3: -10.1369629, -7.5283790, -10.1414614, -7.5154109, -2.0297542, 2.0342009
4: -5.0075369, -2.5014136, -5.0122790, -2.4875448, -2.4014349, 2.3915482
5: -5.4072847, -2.9503319, -5.4231300, -2.9450741, -2.1328802, 2.1731222
6: -13.6888180, -10.7062511, -13.6997213, -10.6703854, -3.0017676, 2.9934702
7: 3.2560339, 5.0214624, 3.2468090, 5.0234723, -1.7072506, 1.7091702
8: -4.4790025, -1.5508962, -4.4856315, -1.5312657, -2.3026419, 2.2726912
9: -2.3410528, 0.1115867, -2.3563299, 0.1154317, -2.3772774, 2.3884530

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0672970
time: 3.92 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0712034
time: 6.33 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317038, -9.2880783, -6.2814522, -2.4163141, 2.4559560
1: -6.8491993, -4.3243461, -6.8198376, -4.3332176, -2.2124343, 2.2401090
2: -8.8176327, -6.4633989, -8.8041668, -6.4693079, -2.1078243, 2.1143439
3: -10.1720533, -7.5003195, -10.1440020, -7.5077829, -2.0747061, 2.0648437
4: -5.0447426, -2.4684753, -5.0150023, -2.4794953, -2.4523978, 2.4602218
5: -5.4397163, -2.9162662, -5.4323187, -2.9420660, -2.1889839, 2.2206140
6: -13.7818356, -10.6415138, -13.7059288, -10.6497040, -3.1321316, 3.0644150
7: 3.2260957, 5.0552692, 3.2414064, 5.0245962, -1.7312896, 1.7486589
8: -4.5310326, -1.5154562, -4.4893522, -1.5199218, -2.3619070, 2.3096972
9: -2.3830662, 0.1482708, -2.3652885, 0.1176077, -2.4209323, 2.4347606

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0672967, upper bound: 1.0711418
time: 4.16 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712037, upper bound: 1.0712036
time: 4.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.37 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0683158
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0642525, upper bound: 1.0683926
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0672970
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0712034
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0672967, upper bound: 1.0711418
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 7, lower bound: -1.0712037, upper bound: 1.0712036

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2149448, -6.2952113, -9.2581244, -6.3018022, -2.3018088, 2.3654618
1: -6.7865696, -4.3618679, -6.8057470, -4.3447609, -2.1284599, 2.1334498
2: -8.7654257, -6.4992485, -8.7869282, -6.4899478, -2.0248742, 2.0449932
3: -10.0935335, -7.5537324, -10.1225395, -7.5375352, -1.9751470, 1.9848502
4: -4.9811821, -2.5230768, -4.9970117, -2.4980164, -2.3654079, 2.3507442
5: -5.3594694, -2.9844561, -5.4002986, -2.9746048, -2.0880475, 2.1152644
6: -13.6640654, -10.7213373, -13.6765709, -10.6763430, -2.9665527, 2.9552336
7: 3.2961416, 5.0177507, 3.2697172, 5.0208259, -1.6628594, 1.6811466
8: -4.4538517, -1.5923243, -4.4749503, -1.5579257, -2.2446933, 2.2376873
9: -2.3186460, 0.0935388, -2.3393304, 0.1064812, -2.3412437, 2.3504047

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0616529
time: 5.73 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
time: 11.63 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2152386, -6.2916164, -9.3029785, -6.2822580, -2.3176026, 2.4022429
1: -6.7868690, -4.3616514, -6.8160534, -4.3406181, -2.1329832, 2.1443276
2: -8.7657528, -6.4971290, -8.8120985, -6.4750786, -2.0518208, 2.0725822
3: -10.0937214, -7.5504727, -10.1594858, -7.5186195, -1.9967470, 2.0259485
4: -4.9822993, -2.5226254, -5.0049419, -2.4859598, -2.3781462, 2.3629446
5: -5.3602314, -2.9799976, -5.4611936, -2.9515865, -2.1079378, 2.1545041
6: -13.6672153, -10.7211180, -13.6978407, -10.6351318, -3.0108271, 2.9726171
7: 3.2947302, 5.0180092, 3.2602415, 5.0373650, -1.6808772, 1.6897700
8: -4.4546108, -1.5902996, -4.5064149, -1.5448527, -2.2676439, 2.2719431
9: -2.3205700, 0.0936959, -2.3524499, 0.1261504, -2.3632503, 2.3673089

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0655506
time: 6.48 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
time: 6.44 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.2741985, -6.2867804, -2.3779707, 2.4099123
1: -6.8252726, -4.3353658, -6.8115201, -4.3370733, -2.1863437, 2.2194276
2: -8.7862673, -6.4962525, -8.7914047, -6.4790239, -2.0687165, 2.0663953
3: -10.1262522, -7.5394750, -10.1258640, -7.5168123, -2.0284500, 2.0050216
4: -5.0140319, -2.4928031, -5.0041666, -2.4881687, -2.4146619, 2.4194911
5: -5.3879519, -2.9686193, -5.4125156, -2.9540749, -2.1575117, 2.1478591
6: -13.7444639, -10.6579409, -13.6951551, -10.6547852, -3.0896788, 3.0319128
7: 3.2723179, 5.0503802, 3.2587042, 5.0229864, -1.6820350, 1.7249547
8: -4.5014772, -1.5659080, -4.4817376, -1.5385866, -2.3077865, 2.2693479
9: -2.3527238, 0.1287827, -2.3558345, 0.1092930, -2.3787680, 2.4017529

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A2_A1_B1

### Relational analysis result of IS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0641728
time: 4.56 seconds

## Relational analysis of IS_A1_A2_A1_B2

### Relational analysis result of IS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0683158
time: 4.85 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -9.3049059, -6.2378545, -9.2744923, -6.2831821, -2.4142604, 2.4257267
1: -6.8355908, -4.3312049, -6.8118210, -4.3368549, -2.1971083, 2.2238188
2: -8.8114977, -6.4812555, -8.7917271, -6.4769163, -2.0964217, 2.0932689
3: -10.1630249, -7.5206203, -10.1260595, -7.5135536, -2.0575199, 2.0266285
4: -5.0219293, -2.4807301, -5.0052900, -2.4877193, -2.4269314, 2.4272141
5: -5.4488506, -2.9456134, -5.4132805, -2.9496183, -2.1964526, 2.1677599
6: -13.7656345, -10.6166916, -13.6982985, -10.6545639, -3.1080408, 3.0762372
7: 3.2628231, 5.0668783, 3.2572956, 5.0232487, -1.6906884, 1.7418494
8: -4.5327868, -1.5527935, -4.4825110, -1.5365648, -2.3195348, 2.2923422
9: -2.3659363, 0.1483328, -2.3577578, 0.1094539, -2.3955564, 2.4229355

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642525, upper bound: 1.0642498
time: 4.50 seconds

## Relational analysis of IS_A1_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642525, upper bound: 1.0683926
time: 4.63 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2472153, -6.2874780, -9.2717247, -6.3000822, -2.3341670, 2.3910708
1: -6.8070784, -4.3519344, -6.8137960, -4.3411102, -2.1511989, 2.1525404
2: -8.7950916, -6.4784746, -8.7993584, -6.4823275, -2.0539393, 2.0758889
3: -10.1367664, -7.5316544, -10.1404800, -7.5315394, -2.0113235, 2.0258174
4: -5.0064049, -2.5018668, -5.0067129, -2.4897883, -2.3954058, 2.3842068
5: -5.4065199, -2.9547989, -5.4193296, -2.9670773, -2.1100588, 2.1644607
6: -13.6856623, -10.7064714, -13.6842060, -10.6714821, -2.9973793, 2.9777346
7: 3.2574492, 5.0212035, 3.2537794, 5.0221734, -1.7042360, 1.7017910
8: -4.4782410, -1.5529308, -4.4818082, -1.5412674, -2.2879233, 2.2625656
9: -2.3391280, 0.1114235, -2.3468599, 0.1146235, -2.3714733, 2.3767014

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0644025
time: 4.11 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0672970
time: 4.09 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838802, -9.3165817, -6.2805305, -2.3499775, 2.4294896
1: -6.8073807, -4.3517170, -6.8241043, -4.3369732, -2.1557255, 2.1636021
2: -8.7954197, -6.4763637, -8.8245163, -6.4674754, -2.0809050, 2.1034613
3: -10.1369638, -7.5283914, -10.1774569, -7.5126023, -2.0328941, 2.0669262
4: -5.0075321, -2.5014157, -5.0146761, -2.4777443, -2.4081707, 2.3963814
5: -5.4072838, -2.9503403, -5.4802051, -2.9440503, -2.1299548, 2.2028255
6: -13.6888123, -10.7062540, -13.7054758, -10.6302738, -3.0416565, 2.9992218
7: 3.2560372, 5.0214629, 3.2443037, 5.0387049, -1.7222371, 1.7104255
8: -4.4790025, -1.5509024, -4.5132599, -1.5281863, -2.3108721, 2.2968931
9: -2.3410499, 0.1115841, -2.3600063, 0.1343278, -2.3934989, 2.3935876

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0682596
time: 6.79 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0712034
time: 6.56 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.2877827, -6.2850561, -2.4107556, 2.4377577
1: -6.8476830, -4.3254142, -6.8195333, -4.3334360, -2.2106690, 2.2387271
2: -8.8160028, -6.4737835, -8.8038387, -6.4714160, -2.0976915, 2.0985520
3: -10.1710596, -7.5164576, -10.1438017, -7.5110598, -2.0663004, 2.0464149
4: -5.0392056, -2.4706733, -5.0138712, -2.4799473, -2.4450917, 2.4540873
5: -5.4359431, -2.9382565, -5.4315519, -2.9465327, -2.1804113, 2.1978157
6: -13.7662449, -10.6425924, -13.7027807, -10.6499233, -3.1163216, 3.0601883
7: 3.2330532, 5.0539637, 3.2428212, 5.0243330, -1.7239246, 1.7456385
8: -4.5272012, -1.5254626, -4.4885788, -1.5219536, -2.3511233, 2.2949972
9: -2.3735833, 0.1474495, -2.3633599, 0.1174446, -2.4092278, 2.4289446

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0641734
time: 4.43 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641742
time: 4.93 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.3378305, -6.2299337, -9.2880783, -6.2814608, -2.4470820, 2.4536052
1: -6.8579874, -4.3212638, -6.8198357, -4.3332171, -2.2215314, 2.2431314
2: -8.8412151, -6.4588223, -8.8041687, -6.4693117, -2.1253929, 2.1253963
3: -10.2079506, -7.4975433, -10.1440010, -7.5077958, -2.0966616, 2.0679793
4: -5.0471773, -2.4586329, -5.0150008, -2.4794965, -2.4571781, 2.4642231
5: -5.4968190, -2.9152360, -5.4323173, -2.9420760, -2.2201672, 2.2177417
6: -13.7874041, -10.6013517, -13.7059240, -10.6497049, -3.1376991, 3.1045723
7: 3.2235560, 5.0704441, 3.2414093, 5.0245953, -1.7325804, 1.7635947
8: -4.5584836, -1.5123401, -4.4893508, -1.5199275, -2.3629603, 2.3179734
9: -2.3868451, 0.1671113, -2.3652856, 0.1176094, -2.4259653, 2.4505091

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0683927, upper bound: 1.0642496
time: 4.65 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0683931, upper bound: 1.0679970
time: 4.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.96 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0616529
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0655506
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
IS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0641728
IS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0683158
IS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0642525, upper bound: 1.0642498
IS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0642525, upper bound: 1.0683926
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0644025
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0672970
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0682596
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0712034
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0641734
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641742
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0683927, upper bound: 1.0642496
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.96
Output dim: 7, lower bound: -1.0683931, upper bound: 1.0679970

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.2149448, -6.2952113, -9.2324219, -6.3033462, -2.2951560, 2.3198311
1: -6.7865696, -4.3618679, -6.7977562, -4.3564796, -2.1002603, 2.1069505
2: -8.7654257, -6.4992485, -8.7813454, -6.4944072, -2.0153131, 2.0252528
3: -10.0935335, -7.5537324, -10.1180573, -7.5506964, -1.9606915, 1.9801280
4: -4.9811821, -2.5230768, -4.9922562, -2.5118928, -2.3415065, 2.3382282
5: -5.3594694, -2.9844561, -5.3844738, -2.9798348, -2.0733142, 2.0912004
6: -13.6640654, -10.7213373, -13.6656504, -10.7122030, -2.9300871, 2.9193645
7: 3.2961416, 5.0177507, 3.2790308, 5.0188293, -1.6611638, 1.6777045
8: -4.4538517, -1.5923243, -4.4683371, -1.5775962, -2.2248554, 2.2297220
9: -2.3186460, 0.0935388, -2.3240569, 0.1026553, -2.3375044, 2.3341250

Time for backsubstitution: 12.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575197, upper bound: 1.0616533
time: 5.46 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0616539
time: 5.49 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.2149448, -6.2952113, -9.2793121, -6.2512522, -2.3421705, 2.3826518
1: -6.7865696, -4.3618679, -6.8396916, -4.3291192, -2.1291256, 2.1540399
2: -8.7654257, -6.4992485, -8.8033876, -6.4813762, -2.0278826, 2.0596039
3: -10.0935335, -7.5537324, -10.1529608, -7.5225863, -1.9895296, 2.0165162
4: -4.9811821, -2.5230768, -5.0294862, -2.4788394, -2.3758531, 2.3747463
5: -5.3594694, -2.9844561, -5.4168482, -2.9458468, -2.1083622, 2.1240034
6: -13.6640654, -10.7213373, -13.7586060, -10.6474819, -2.9955540, 3.0343003
7: 3.2961416, 5.0177507, 3.2488728, 5.0525904, -1.6955156, 1.7033081
8: -4.4538517, -1.5923243, -4.5204101, -1.5420923, -2.2608151, 2.2841172
9: -2.3186460, 0.0935388, -2.3661029, 0.1393178, -2.3745604, 2.3767018

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
time: 4.64 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
time: 5.16 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.2152386, -6.2916164, -9.2772512, -6.2837772, -2.3109689, 2.3650842
1: -6.7868690, -4.3616514, -6.8080626, -4.3523951, -2.1047621, 2.1177909
2: -8.7657528, -6.4971290, -8.8065348, -6.4795561, -2.0422325, 2.0528812
3: -10.0937214, -7.5504727, -10.1549950, -7.5318170, -1.9823339, 2.0211792
4: -4.9822993, -2.5226254, -5.0001531, -2.4998963, -2.3541842, 2.3504519
5: -5.3602314, -2.9799976, -5.4453173, -2.9568315, -2.0932207, 2.1344481
6: -13.6672153, -10.7211180, -13.6870728, -10.6710453, -2.9743080, 2.9364996
7: 3.2947302, 5.0180092, 3.2695661, 5.0353632, -1.6791739, 1.6863323
8: -4.4546108, -1.5902996, -4.4997439, -1.5645065, -2.2477989, 2.2639086
9: -2.3205700, 0.0936959, -2.3371677, 0.1223143, -2.3595004, 2.3510268

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0614143
time: 6.50 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0655506
time: 6.57 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.2152386, -6.2916164, -9.3241959, -6.2317171, -2.3580046, 2.4076860
1: -6.7868690, -4.3616514, -6.8499937, -4.3249650, -2.1337228, 2.1649048
2: -8.7657528, -6.4971290, -8.8286057, -6.4663949, -2.0549498, 2.0872893
3: -10.0937214, -7.5504727, -10.1898174, -7.5036936, -2.0111246, 2.0392737
4: -4.9822993, -2.5226254, -5.0374231, -2.4667864, -2.3886113, 2.3839436
5: -5.3602314, -2.9799976, -5.4777431, -2.9228301, -2.1282597, 2.1591172
6: -13.6672153, -10.7211180, -13.7797680, -10.6062393, -3.0333099, 3.0519338
7: 3.2947302, 5.0180092, 3.2393785, 5.0690775, -1.7134719, 1.7119632
8: -4.4546108, -1.5902996, -4.5517044, -1.5289764, -2.2837896, 2.2959032
9: -2.3205700, 0.0936959, -2.3793383, 0.1589456, -2.3938203, 2.3936071

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0642496
time: 6.65 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683925
time: 6.62 seconds

## BFS IS instance: IS_A1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.2555475, -6.2928309, -2.3705225, 2.3939176
1: -6.8252726, -4.3353658, -6.7972393, -4.3432703, -2.1805096, 2.2047262
2: -8.7862673, -6.4962525, -8.7741823, -6.4931536, -2.0548491, 2.0489314
3: -10.1262522, -7.5394750, -10.0990248, -7.5330491, -2.0115099, 1.9769900
4: -5.0140319, -2.4928031, -4.9887342, -2.5018764, -2.3962030, 2.4039254
5: -5.3879519, -2.9686193, -5.3838229, -2.9762769, -2.1329169, 2.1191363
6: -13.7444639, -10.6579409, -13.6812086, -10.6649380, -3.0739107, 3.0147815
7: 3.2723179, 5.0503802, 3.2815204, 5.0208874, -1.6799259, 1.7018690
8: -4.5014772, -1.5659080, -4.4631228, -1.5611744, -2.2922430, 2.2465959
9: -2.3527238, 0.1287827, -2.3427229, 0.0987484, -2.3664765, 2.3880324

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0603224
time: 4.79 seconds

## Relational analysis of IS_A1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0641728
time: 4.70 seconds

## BFS IS instance: IS_A1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.2877722, -6.2850599, -2.3816481, 2.4187770
1: -6.8252726, -4.3353658, -6.8195300, -4.3334394, -2.1902709, 2.2252994
2: -8.7862673, -6.4962525, -8.8038244, -6.4714212, -2.0740428, 2.0787351
3: -10.1262522, -7.5394750, -10.1437912, -7.5110636, -2.0338278, 2.0236912
4: -5.0140319, -2.4928031, -5.0138636, -2.4799521, -2.4200788, 2.4236405
5: -5.3879519, -2.9686193, -5.4315443, -2.9465384, -2.1627474, 2.1599112
6: -13.7444639, -10.6579409, -13.7027750, -10.6499290, -3.0880342, 3.0401807
7: 3.2723179, 5.0503802, 3.2428331, 5.0243325, -1.6834292, 1.7417210
8: -4.5014772, -1.5659080, -4.4885764, -1.5219693, -2.3136172, 2.2753437
9: -2.3527238, 0.1287827, -2.3633533, 0.1174407, -2.3883095, 2.4087572

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0644653
time: 4.72 seconds

## Relational analysis of IS_A1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0683158
time: 4.79 seconds

## BFS IS instance: IS_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3049059, -6.2378545, -9.2558384, -6.2892361, -2.4065671, 2.4097311
1: -6.8355908, -4.3312049, -6.7975388, -4.3430529, -2.1912751, 2.2091157
2: -8.8114977, -6.4812555, -8.7745056, -6.4910402, -2.0825543, 2.0758042
3: -10.1630249, -7.5206203, -10.0992174, -7.5297871, -2.0404081, 1.9986019
4: -5.0219293, -2.4807301, -4.9898515, -2.5014248, -2.4083996, 2.4158950
5: -5.4488506, -2.9456134, -5.3845844, -2.9718184, -2.1714644, 2.1390324
6: -13.7656345, -10.6166916, -13.6843519, -10.6647139, -3.0915260, 3.0591044
7: 3.2628231, 5.0668783, 3.2801104, 5.0211515, -1.6885781, 1.7198400
8: -4.5327868, -1.5527935, -4.4638920, -1.5591545, -2.3039894, 2.2695906
9: -2.3659363, 0.1483328, -2.3446484, 0.0989083, -2.3832664, 2.4093909

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A1_A2_A2_B1_A1

### Relational analysis result of IS_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0517576, upper bound: 1.0634297
time: 4.99 seconds

## Relational analysis of IS_A1_A2_A2_B1_A2

### Relational analysis result of IS_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0634292, upper bound: 1.0634276
time: 4.34 seconds

## BFS IS instance: IS_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3049059, -6.2378545, -9.2880669, -6.2814641, -2.4158664, 2.4345915
1: -6.8355908, -4.3312049, -6.8198342, -4.3332205, -2.2010384, 2.2296925
2: -8.8114977, -6.4812555, -8.8041534, -6.4693146, -2.1017437, 2.1056092
3: -10.1630249, -7.5206203, -10.1439886, -7.5078001, -2.0577559, 2.0452962
4: -5.0219293, -2.4807301, -5.0149908, -2.4795015, -2.4306102, 2.4313602
5: -5.4488506, -2.9456134, -5.4323096, -2.9420815, -2.1899629, 2.1798198
6: -13.7656345, -10.6166916, -13.7059193, -10.6497078, -3.1056490, 3.0845051
7: 3.2628231, 5.0668783, 3.2414193, 5.0245948, -1.6920793, 1.7531787
8: -4.5327868, -1.5527935, -4.4893479, -1.5199461, -2.3253646, 2.2983410
9: -2.3659363, 0.1483328, -2.3652782, 0.1176049, -2.4050975, 2.4291329

Time for backsubstitution: 12.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A1_A2_A2_B2_A1

### Relational analysis result of IS_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0517576, upper bound: 1.0675733
time: 4.42 seconds

## Relational analysis of IS_A1_A2_A2_B2_A2

### Relational analysis result of IS_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0634292, upper bound: 1.0675712
time: 4.36 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.2472153, -6.2874780, -9.2460518, -6.3016319, -2.3275523, 2.3458743
1: -6.8070784, -4.3519344, -6.8058796, -4.3528051, -2.1230869, 2.1258912
2: -8.7950916, -6.4784746, -8.7937918, -6.4867725, -2.0444202, 2.0561080
3: -10.1367664, -7.5316544, -10.1360054, -7.5445008, -1.9968681, 2.0211506
4: -5.0064049, -2.5018668, -5.0019770, -2.5036511, -2.3714871, 2.3715725
5: -5.4065199, -2.9547989, -5.4034939, -2.9723299, -2.0954762, 2.1403863
6: -13.6856623, -10.7064714, -13.6732817, -10.7073393, -2.9609585, 2.9668102
7: 3.2574492, 5.0212035, 3.2629972, 5.0201750, -1.7025442, 1.6984642
8: -4.4782410, -1.5529308, -4.4752235, -1.5608931, -2.2680764, 2.2545466
9: -2.3391280, 0.1114235, -2.3315833, 0.1107912, -2.3678207, 2.3605409

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644025, upper bound: 1.0644017
time: 7.49 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644025, upper bound: 1.0644050
time: 4.45 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.2472153, -6.2874780, -9.2929230, -6.2495356, -2.3744211, 2.4082494
1: -6.8070784, -4.3519344, -6.8476272, -4.3254733, -2.1519747, 2.1730626
2: -8.7950916, -6.4784746, -8.8157768, -6.4738135, -2.0569739, 2.0904226
3: -10.1367664, -7.5316544, -10.1708908, -7.5164642, -2.0256929, 2.0574660
4: -5.0064049, -2.5018668, -5.0391550, -2.4706907, -2.4059725, 2.4090819
5: -5.4065199, -2.9547989, -5.4358959, -2.9383163, -2.1304407, 2.1732600
6: -13.6856623, -10.7064714, -13.7662048, -10.6426678, -3.0264397, 3.0597334
7: 3.2574492, 5.0212035, 3.2330499, 5.0539484, -1.7369080, 1.7238266
8: -4.4782410, -1.5529308, -4.5272002, -1.5254817, -2.3040109, 2.3094616
9: -2.3391280, 0.1114235, -2.3735433, 0.1474400, -2.4047408, 2.4031029

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0654862, upper bound: 1.0603228
time: 5.72 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0654866, upper bound: 1.0641384
time: 5.30 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838802, -9.2908783, -6.2820549, -2.3433838, 2.3923876
1: -6.8073807, -4.3517170, -6.8161888, -4.3487253, -2.1275783, 2.1369319
2: -8.7954197, -6.4763637, -8.8189678, -6.4719377, -2.0713596, 2.0836990
3: -10.1369638, -7.5283914, -10.1729765, -7.5255857, -2.0184703, 2.0622044
4: -5.0075321, -2.5014157, -5.0099087, -2.4916680, -2.3842049, 2.3837686
5: -5.4072838, -2.9503403, -5.4643192, -2.9493203, -2.1153870, 2.1827481
6: -13.6888123, -10.7062540, -13.6947021, -10.6661816, -3.0051832, 2.9840627
7: 3.2560372, 5.0214629, 3.2535300, 5.0367031, -1.7205362, 1.7070911
8: -4.4790025, -1.5509024, -4.5066204, -1.5477939, -2.2910180, 2.2888207
9: -2.3410499, 0.1115841, -2.3447208, 0.1304876, -2.3898339, 2.3774261

Time for backsubstitution: 12.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5751

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682480, upper bound: 1.0672596
time: 4.99 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682480, upper bound: 1.0682500
time: 4.72 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838802, -9.3378048, -6.2299900, -2.3902724, 2.4349811
1: -6.8073807, -4.3517170, -6.8579330, -4.3213239, -2.1565733, 2.1840997
2: -8.7954197, -6.4763637, -8.8409863, -6.4588509, -2.0840592, 2.1181083
3: -10.1369638, -7.5283914, -10.2077799, -7.4975510, -2.0472641, 2.0824389
4: -5.0075321, -2.5014157, -5.0471258, -2.4586511, -2.4187918, 2.4197996
5: -5.4072838, -2.9503403, -5.4967728, -2.9152918, -2.1503515, 2.2074950
6: -13.6888123, -10.7062540, -13.7873669, -10.6014290, -3.0627251, 3.0811129
7: 3.2560372, 5.0214629, 3.2235527, 5.0704279, -1.7548556, 1.7324804
8: -4.4790025, -1.5509024, -4.5584812, -1.5123610, -2.3269844, 2.3215365
9: -2.3410499, 0.1115841, -2.3868036, 0.1671035, -2.4257870, 2.4199901

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0655509, upper bound: 1.0642499
time: 9.39 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0655513, upper bound: 1.0642498
time: 4.76 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.2555475, -6.2928309, -2.4053779, 2.4037418
1: -6.8476830, -4.3254142, -6.7972393, -4.3432703, -2.2009048, 2.2146730
2: -8.8160028, -6.4737835, -8.7741823, -6.4931536, -2.0845981, 2.0687377
3: -10.1710596, -7.5164576, -10.0990248, -7.5330491, -2.0582366, 1.9997051
4: -5.0392056, -2.4706733, -4.9887342, -2.5018764, -2.4151640, 2.4280715
5: -5.4359431, -2.9382565, -5.3838229, -2.9762769, -2.1770139, 2.1495132
6: -13.7662449, -10.6425924, -13.6812086, -10.6649380, -3.0955505, 3.0332131
7: 3.2330532, 5.0539637, 3.2815204, 5.0208874, -1.7204199, 1.7055144
8: -4.5272012, -1.5254626, -4.4631228, -1.5611744, -2.3133640, 2.2839828
9: -2.3735833, 0.1474495, -2.3427229, 0.0987484, -2.3873916, 2.4098206

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0603230
time: 4.31 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0641734
time: 4.56 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.2877722, -6.2850599, -2.4107523, 2.4341674
1: -6.8476830, -4.3254142, -6.8195300, -4.3334394, -2.2142978, 2.2387221
2: -8.8160028, -6.4737835, -8.8038244, -6.4714212, -2.0976877, 2.0925219
3: -10.1710596, -7.5164576, -10.1437912, -7.5110636, -2.0662971, 2.0321474
4: -5.0392056, -2.4706733, -5.0138636, -2.4799521, -2.4450841, 2.4530735
5: -5.4359431, -2.9382565, -5.4315443, -2.9465384, -2.1804028, 2.1669998
6: -13.7662449, -10.6425924, -13.7027750, -10.6499290, -3.1163158, 3.0601826
7: 3.2330532, 5.0539637, 3.2428331, 5.0243325, -1.7242265, 1.7456290
8: -4.5272012, -1.5254626, -4.4885764, -1.5219693, -2.3409910, 2.2949905
9: -2.3735833, 0.1474495, -2.3633533, 0.1174407, -2.4079432, 2.4289351

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641396
time: 5.35 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641741
time: 5.46 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3378305, -6.2299337, -9.2558384, -6.2892361, -2.4321041, 2.4195874
1: -6.8579874, -4.3212638, -6.7975388, -4.3430529, -2.2117662, 2.2190664
2: -8.8412151, -6.4588223, -8.7745056, -6.4910402, -2.1122942, 2.0955799
3: -10.2079506, -7.4975433, -10.0992174, -7.5297871, -2.0715513, 2.0212755
4: -5.0471773, -2.4586329, -4.9898515, -2.5014248, -2.4243851, 2.4384234
5: -5.4968190, -2.9152360, -5.3845844, -2.9718184, -2.1948910, 2.1694283
6: -13.7874041, -10.6013517, -13.6843519, -10.6647139, -3.1131678, 3.0775404
7: 3.2235560, 5.0704441, 3.2801104, 5.0211515, -1.7290778, 1.7234602
8: -4.5584836, -1.5123401, -4.4638920, -1.5591545, -2.3251972, 2.3069386
9: -2.3868451, 0.1671113, -2.3446484, 0.0989083, -2.4041314, 2.4250343

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A2_A2_A2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0558900, upper bound: 1.0634264
time: 4.44 seconds

## Relational analysis of IS_A2_A2_A2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0675703, upper bound: 1.0634276
time: 4.49 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.3378305, -6.2299337, -9.2880669, -6.2814641, -2.4470615, 2.4500148
1: -6.8579874, -4.3212638, -6.8198342, -4.3332205, -2.2253423, 2.2431262
2: -8.8412151, -6.4588223, -8.8041534, -6.4693146, -2.1253886, 2.1194074
3: -10.2079506, -7.4975433, -10.1439886, -7.5078001, -2.0965714, 2.0537121
4: -5.0471773, -2.4586329, -5.0149908, -2.4795015, -2.4571724, 2.4647903
5: -5.4968190, -2.9152360, -5.4323096, -2.9420815, -2.2201614, 2.1869149
6: -13.7874041, -10.6013517, -13.7059193, -10.6497078, -3.1376963, 3.1045675
7: 3.2235560, 5.0704441, 3.2414193, 5.0245948, -1.7328823, 1.7635849
8: -4.5584836, -1.5123401, -4.4893479, -1.5199461, -2.3529024, 2.3179662
9: -2.3868451, 0.1671113, -2.3652782, 0.1176049, -2.4246931, 2.4505012

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A2_A2_A2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0558903, upper bound: 1.0634271
time: 5.78 seconds

## Relational analysis of IS_A2_A2_A2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0675706, upper bound: 1.0672961
time: 4.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.86 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0575197, upper bound: 1.0616533
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0616539
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0614143
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0655506
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0642496
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683925
IS_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0603224
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0641728
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0644653
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0683158
IS_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0517576, upper bound: 1.0634297
IS_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0634292, upper bound: 1.0634276
IS_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0517576, upper bound: 1.0675733
IS_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0634292, upper bound: 1.0675712
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0644025, upper bound: 1.0644017
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0644025, upper bound: 1.0644050
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0654862, upper bound: 1.0603228
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0654866, upper bound: 1.0641384
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0682480, upper bound: 1.0672596
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0682480, upper bound: 1.0682500
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0655509, upper bound: 1.0642499
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0655513, upper bound: 1.0642498
IS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0603230
IS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0641734
IS_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641396
IS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641741
IS_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0558900, upper bound: 1.0634264
IS_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0675703, upper bound: 1.0634276
IS_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0558903, upper bound: 1.0634271
IS_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.86
Output dim: 7, lower bound: -1.0675706, upper bound: 1.0672961

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2137794, -6.3093643, -9.2324219, -6.3033462, -2.2936072, 2.3056781
1: -6.7853842, -4.3627400, -6.7977562, -4.3564796, -2.0990491, 2.1061182
2: -8.7641144, -6.5075798, -8.7813454, -6.4944072, -2.0095453, 2.0138233
3: -10.0927734, -7.5665693, -10.1180573, -7.5506964, -1.9567986, 1.9662251
4: -4.9767933, -2.5248652, -4.9922562, -2.5118928, -2.3364539, 2.3344088
5: -5.3564520, -3.0019968, -5.3844738, -2.9798348, -2.0700150, 2.0736737
6: -13.6516829, -10.7222109, -13.6656504, -10.7122030, -2.9174099, 2.9184484
7: 3.3016930, 5.0167184, 3.2790308, 5.0188293, -1.6555352, 1.6764420
8: -4.4508424, -1.6002913, -4.4683371, -1.5775962, -2.2189360, 2.2192125
9: -2.3110857, 0.0929170, -2.3240569, 0.1026553, -2.3287416, 2.3312654

Time for backsubstitution: 12.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575177, upper bound: 1.0575177
time: 4.59 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575199, upper bound: 1.0616531
time: 5.97 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -9.2585220, -6.2898102, -9.2324219, -6.3033462, -2.3381333, 2.3250117
1: -6.7927766, -4.3586483, -6.7977562, -4.3564796, -2.1070600, 2.1102231
2: -8.7893171, -6.4939280, -8.7813454, -6.4944072, -2.0343003, 2.0276599
3: -10.1274815, -7.5477257, -10.1180573, -7.5506964, -1.9918213, 1.9848526
4: -4.9846506, -2.5138705, -4.9922562, -2.5118928, -2.3441443, 2.3447523
5: -5.4164286, -2.9790025, -5.3844738, -2.9798348, -2.1186266, 2.0967178
6: -13.6731148, -10.6812363, -13.6656504, -10.7122030, -2.9379072, 2.9592671
7: 3.2927065, 5.0332623, 3.2790308, 5.0188293, -1.6644101, 1.6930271
8: -4.4805675, -1.5871987, -4.4683371, -1.5775962, -2.2470922, 2.2325788
9: -2.3241744, 0.1114789, -2.3240569, 0.1026553, -2.3421168, 2.3495736

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575177, upper bound: 1.0575177
time: 4.89 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575199, upper bound: 1.0616553
time: 5.68 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 23.55 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 23.55
Output dim: 7, lower bound: -1.0575177, upper bound: 1.0575177
IS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 23.55
Output dim: 7, lower bound: -1.0575199, upper bound: 1.0616531
IS_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 23.55
Output dim: 7, lower bound: -1.0575177, upper bound: 1.0575177
IS_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 23.55
Output dim: 7, lower bound: -1.0575199, upper bound: 1.0616553
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0614143
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0655506
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0642496
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683925
IS_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0603224
IS_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0641728
IS_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0644653
IS_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0603253, upper bound: 1.0683158
IS_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0517576, upper bound: 1.0634297
IS_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0634292, upper bound: 1.0634276
IS_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0517576, upper bound: 1.0675733
IS_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0634292, upper bound: 1.0675712
IS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0644025, upper bound: 1.0644017
IS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0644025, upper bound: 1.0644050
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0654862, upper bound: 1.0603228
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0654866, upper bound: 1.0641384
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0682480, upper bound: 1.0672596
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0682480, upper bound: 1.0682500
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0655509, upper bound: 1.0642499
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0655513, upper bound: 1.0642498
IS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0603230
IS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0641734
IS_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641396
IS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641741
IS_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0558900, upper bound: 1.0634264
IS_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0675703, upper bound: 1.0634276
IS_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0558903, upper bound: 1.0634271
IS_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 23.55
Output dim: 7, lower bound: -1.0675706, upper bound: 1.0672961
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.7166380882263184
rel_dist={7: [-1.071222731717584, 1.071219330481989]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9651484, upper bound: 0.9684750
time: 4.43 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703681, upper bound: 0.9703673
time: 4.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.53 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.53
Output dim: 7, lower bound: -0.9651484, upper bound: 0.9684750
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.53
Output dim: 7, lower bound: -0.9703681, upper bound: 0.9703673

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.2558632, -6.2892208, -9.2720938, -6.2834878, -2.2849650, 2.2950592
1: -6.7975464, -4.3430395, -6.8103442, -4.3374972, -2.1080170, 2.1159763
2: -8.7745104, -6.4910307, -8.7895060, -6.4783158, -1.9981623, 2.0007675
3: -10.0992203, -7.5297594, -10.1228170, -7.5145760, -1.9204283, 1.9291086
4: -4.9898586, -2.5014129, -5.0035419, -2.4892335, -2.3380957, 2.3351412
5: -5.3845930, -2.9718044, -5.4098282, -2.9510093, -2.0747633, 2.0767848
6: -13.6843634, -10.6646891, -13.6969309, -10.6554356, -2.9405022, 2.9411159
7: 3.2800965, 5.0211525, 3.2601409, 5.0230069, -1.6272633, 1.6454487
8: -4.4638987, -1.5591264, -4.4812579, -1.5395350, -2.1858640, 2.1889603
9: -2.3446670, 0.0989108, -2.3563974, 0.1079956, -2.3209844, 2.3227434

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625205, upper bound: 0.9684654
time: 4.27 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9651427, upper bound: 0.9684653
time: 4.57 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.2880926, -6.2814493, -9.2881021, -6.2814484, -2.3173041, 2.3233652
1: -6.8198409, -4.3332071, -6.8198442, -4.3332047, -2.1330109, 2.1363935
2: -8.8041601, -6.4693065, -8.8041706, -6.4693027, -2.0278797, 2.0345206
3: -10.1439934, -7.5077729, -10.1440029, -7.5077686, -1.9584746, 1.9734855
4: -5.0149994, -2.4794898, -5.0150065, -2.4794867, -2.3691359, 2.3713446
5: -5.4323173, -2.9420679, -5.4323225, -2.9420624, -2.0967054, 2.1297421
6: -13.7059288, -10.6496811, -13.7059336, -10.6496763, -2.9716344, 2.9892979
7: 3.2414083, 5.0245953, 3.2413979, 5.0245967, -1.6687293, 1.6690170
8: -4.4893527, -1.5199199, -4.4893570, -1.5199046, -2.2330914, 2.2140222
9: -2.3652968, 0.1176064, -2.3653016, 0.1176112, -2.3529582, 2.3511825

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677206, upper bound: 0.9703605
time: 4.62 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703619, upper bound: 0.9703642
time: 4.94 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 22.23 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 7, lower bound: -0.9625205, upper bound: 0.9684654
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 7, lower bound: -0.9651427, upper bound: 0.9684653
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 7, lower bound: -0.9677206, upper bound: 0.9703605
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 22.23
Output dim: 7, lower bound: -0.9703619, upper bound: 0.9703642

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916050, -9.2531385, -6.2845945, -2.2161846, 2.2611938
1: -6.7868700, -4.3616509, -6.8045034, -4.3461647, -2.0537119, 2.0588889
2: -8.7657557, -6.4971232, -8.7854567, -6.4816551, -1.9690971, 1.9824083
3: -10.0937233, -7.5504613, -10.1195745, -7.5246177, -1.9041040, 1.9028199
4: -4.9823031, -2.5226245, -5.0000553, -2.4994802, -2.3008962, 2.2892013
5: -5.3602338, -2.9799883, -5.3981647, -2.9548299, -2.0284715, 2.0383120
6: -13.6672230, -10.7211151, -13.6890411, -10.6817446, -2.8743143, 2.8788252
7: 3.2947273, 5.0180111, 3.2670918, 5.0215750, -1.6174154, 1.6362151
8: -4.4546123, -1.5902944, -4.4764829, -1.5540061, -2.1611037, 2.1527870
9: -2.3205733, 0.0936966, -2.3450265, 0.1052258, -2.2918863, 2.3053799

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9624946, upper bound: 0.9655598
time: 4.86 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
time: 4.39 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396040, -9.2720652, -6.2834911, -2.2760921, 2.3215218
1: -6.8267632, -4.3343019, -6.8103361, -4.3375115, -2.1160560, 2.1443250
2: -8.7878790, -6.4858274, -8.7895012, -6.4783201, -2.0083222, 2.0100677
3: -10.1272221, -7.5233526, -10.1228123, -7.5145912, -1.9496000, 1.9335866
4: -5.0195088, -2.4906089, -5.0035367, -2.4892461, -2.3551326, 2.3551350
5: -5.3917122, -2.9466255, -5.4098225, -2.9510145, -2.0829396, 2.0883553
6: -13.7600527, -10.6568575, -13.6969280, -10.6554642, -3.0123024, 2.9438000
7: 3.2653699, 5.0516911, 3.2601542, 5.0230060, -1.6408181, 1.6770743
8: -4.5052948, -1.5559216, -4.4812508, -1.5395575, -2.2220631, 2.1894519
9: -2.3622165, 0.1295742, -2.3563812, 0.1079929, -2.3357639, 2.3534555

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6208

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_A1

### Relational analysis result of IS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9622329, upper bound: 0.9684327
time: 5.45 seconds

## Relational analysis of IS_A1_A2_A2

### Relational analysis result of IS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9651406, upper bound: 0.9684630
time: 5.19 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -9.2475052, -6.2838678, -9.2691689, -6.2825584, -2.2489667, 2.2894073
1: -6.8073835, -4.3517179, -6.8140683, -4.3418493, -2.0769429, 2.0790546
2: -8.7954216, -6.4763594, -8.8001165, -6.4726272, -1.9988704, 2.0155158
3: -10.1369629, -7.5283790, -10.1407671, -7.5174718, -1.9405107, 1.9471593
4: -5.0075369, -2.5014136, -5.0115356, -2.4897213, -2.3318591, 2.3245487
5: -5.4072847, -2.9503319, -5.4206467, -2.9459038, -2.0498738, 2.0910912
6: -13.6888180, -10.7062511, -13.6980343, -10.6759834, -2.9055338, 2.9269466
7: 3.2560339, 5.0214624, 3.2482662, 5.0231638, -1.6590066, 1.6598804
8: -4.4790025, -1.5508962, -4.4846087, -1.5343375, -2.2072086, 2.1777749
9: -2.3410528, 0.1115867, -2.3539338, 0.1148349, -2.3238254, 2.3329344

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677165, upper bound: 0.9674689
time: 4.78 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677188, upper bound: 0.9703585
time: 4.49 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317038, -9.2880707, -6.2814493, -2.3092675, 2.3517380
1: -6.8491993, -4.3243461, -6.8198366, -4.3332195, -2.1409121, 2.1647213
2: -8.8176327, -6.4633989, -8.8041649, -6.4693069, -2.0380011, 2.0444221
3: -10.1720533, -7.5003195, -10.1439981, -7.5077872, -1.9876573, 1.9783573
4: -5.0447426, -2.4684753, -5.0150003, -2.4794977, -2.3865719, 2.3922238
5: -5.4397163, -2.9162662, -5.4323158, -2.9420676, -2.1052580, 2.1418586
6: -13.7818356, -10.6415138, -13.7059288, -10.6497097, -3.0422926, 2.9929676
7: 3.2260957, 5.0552692, 3.2414103, 5.0245953, -1.6829326, 1.7007376
8: -4.5310326, -1.5154562, -4.4893503, -1.5199280, -2.2678561, 2.2152011
9: -2.3830662, 0.1482708, -2.3652844, 0.1176065, -2.3679008, 2.3818402

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9674689, upper bound: 0.9703414
time: 6.38 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703598, upper bound: 0.9703624
time: 5.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.36 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9624946, upper bound: 0.9655598
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
IS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9622329, upper bound: 0.9684327
IS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9651406, upper bound: 0.9684630
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9677165, upper bound: 0.9674689
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9677188, upper bound: 0.9703585
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9674689, upper bound: 0.9703414
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.36
Output dim: 7, lower bound: -0.9703598, upper bound: 0.9703624

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2152348, -6.2916193, -9.2965164, -6.2828097, -2.2116532, 2.2909310
1: -6.7868690, -4.3616519, -6.8133144, -4.3431215, -2.0568628, 2.0680072
2: -8.7657528, -6.4971304, -8.8090029, -6.4771967, -1.9782896, 1.9998581
3: -10.0937233, -7.5504746, -10.1555376, -7.5218072, -1.9050386, 1.9350020
4: -4.9822993, -2.5226252, -5.0024309, -2.4896705, -2.3076067, 2.2934289
5: -5.3602304, -2.9800000, -5.4552531, -2.9538202, -2.0233278, 2.0672295
6: -13.6672163, -10.7211161, -13.6947813, -10.6416349, -2.9142046, 2.8778143
7: 3.2947311, 5.0180087, 3.2645855, 5.0368137, -1.6324129, 1.6368779
8: -4.4546089, -1.5903029, -4.5041208, -1.5509300, -2.1675324, 2.1769176
9: -2.3205693, 0.0936962, -2.3486702, 0.1240889, -2.3080702, 2.3092775

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9658417
time: 4.15 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
time: 4.32 seconds

## BFS IS instance: IS_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.2715816, -6.2893944, -2.2682509, 2.3030920
1: -6.8252726, -4.3353658, -6.8098421, -4.3378711, -2.1141963, 2.1427407
2: -8.7862673, -6.4962525, -8.7889652, -6.4817777, -1.9963603, 1.9933233
3: -10.1262522, -7.5394750, -10.1224899, -7.5199466, -1.9389176, 1.9145243
4: -5.0140319, -2.4928031, -5.0016928, -2.4899881, -2.3472447, 2.3478839
5: -5.3879519, -2.9686193, -5.4085646, -2.9583278, -2.0715790, 2.0650387
6: -13.7444639, -10.6579409, -13.6917725, -10.6558247, -2.9961295, 2.9373679
7: 3.2723179, 5.0503802, 3.2624679, 5.0225754, -1.6332510, 1.6731336
8: -4.5014772, -1.5659080, -4.4799805, -1.5428791, -2.2094245, 2.1737795
9: -2.3527238, 0.1287827, -2.3532267, 0.1077271, -2.3235555, 2.3462024

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6208

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A1_A2_A1_A1

### Relational analysis result of IS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9511470, upper bound: 0.9680033
time: 4.92 seconds

## Relational analysis of IS_A1_A2_A1_A2

### Relational analysis result of IS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9617985, upper bound: 0.9680053
time: 4.24 seconds

## BFS IS instance: IS_A1_A2_A2

### Backsubstitution after applying IS history:
0: -9.3049059, -6.2378545, -9.2720642, -6.2835073, -2.3051670, 2.3169553
1: -6.8355908, -4.3312049, -6.8103366, -4.3375130, -2.1251006, 2.1473248
2: -8.8114977, -6.4812555, -8.7895002, -6.4783254, -2.0259190, 2.0192125
3: -10.1630249, -7.5206203, -10.1228132, -7.5146084, -1.9676142, 1.9345272
4: -5.0219293, -2.4807301, -5.0035310, -2.4892473, -2.3584914, 2.3564925
5: -5.4488506, -2.9456134, -5.4098206, -2.9510260, -2.1110229, 2.0832343
6: -13.7656345, -10.6166916, -13.6969194, -10.6554642, -3.0118504, 2.9837537
7: 3.2628231, 5.0668783, 3.2601576, 5.0230055, -1.6415272, 1.6897109
8: -4.5327868, -1.5527935, -4.4812474, -1.5395660, -2.2231035, 2.1959274
9: -2.3659363, 0.1483328, -2.3563776, 0.1079926, -2.3395271, 2.3670740

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A1_A2_A2_A1

### Relational analysis result of IS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9540333, upper bound: 0.9680361
time: 5.13 seconds

## Relational analysis of IS_A1_A2_A2_A2

### Relational analysis result of IS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9647101, upper bound: 0.9680371
time: 4.37 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2470255, -6.2897749, -9.2676964, -6.3003211, -2.2305689, 2.2815504
1: -6.8068833, -4.3520746, -6.8125610, -4.3429441, -2.0753698, 2.0771685
2: -8.7948799, -6.4798236, -8.7984867, -6.4830303, -1.9820762, 2.0035164
3: -10.1366463, -7.5337410, -10.1397858, -7.5335975, -1.9214563, 1.9365201
4: -5.0056829, -2.5021553, -5.0059700, -2.4919624, -2.3250079, 2.3165846
5: -5.4060316, -2.9576426, -5.4168453, -2.9679077, -2.0265269, 2.0795884
6: -13.6836510, -10.7066107, -13.6824961, -10.6770821, -2.8990908, 2.9106531
7: 3.2583513, 5.0210376, 3.2552376, 5.0218663, -1.6550796, 1.6522992
8: -4.4777541, -1.5542264, -4.4807930, -1.5443420, -2.1915226, 2.1659451
9: -2.3379030, 0.1113198, -2.3444614, 0.1140275, -2.3165998, 2.3207178

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 874

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673532, upper bound: 0.9562887
time: 4.67 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673532, upper bound: 0.9670878
time: 4.66 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2475061, -6.2838831, -9.3125477, -6.2807631, -2.2444406, 2.3205338
1: -6.8073816, -4.3517175, -6.8228703, -4.3388166, -2.0800877, 2.0883541
2: -8.7954206, -6.4763641, -8.8236475, -6.4681816, -2.0080523, 2.0329423
3: -10.1369629, -7.5283947, -10.1767645, -7.5146642, -1.9414268, 1.9798822
4: -5.0075331, -2.5014160, -5.0139279, -2.4799275, -2.3385859, 2.3287416
5: -5.4072819, -2.9503427, -5.4777164, -2.9448845, -2.0447197, 2.1185277
6: -13.6888123, -10.7062511, -13.7037830, -10.6358786, -2.9454136, 2.9259133
7: 3.2560368, 5.0214620, 3.2457614, 5.0383968, -1.6739917, 1.6605525
8: -4.4790010, -1.5509028, -4.5122347, -1.5312576, -2.2136288, 2.2019658
9: -2.3410478, 0.1115847, -2.3576081, 0.1337305, -2.3400426, 2.3367863

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 874

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9591746
time: 5.22 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9699971
time: 4.51 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.2875881, -6.2873545, -2.3014169, 2.3332887
1: -6.8476830, -4.3254142, -6.8193369, -4.3335795, -2.1390066, 2.1631398
2: -8.8160028, -6.4737835, -8.8036251, -6.4727635, -2.0260100, 2.0276916
3: -10.1710596, -7.5164576, -10.1436710, -7.5131507, -1.9769945, 1.9593019
4: -5.0392056, -2.4706733, -5.0131478, -2.4802396, -2.3786507, 2.3852673
5: -5.4359431, -2.9382565, -5.4310584, -2.9493804, -2.0938392, 2.1185322
6: -13.7662449, -10.6425924, -13.7007732, -10.6500711, -3.0261183, 2.9865236
7: 3.2330532, 5.0539637, 3.2437263, 5.0241642, -1.6753631, 1.6968043
8: -4.5272012, -1.5254626, -4.4880819, -1.5232520, -2.2551413, 2.1995344
9: -2.3735833, 0.1474495, -2.3621292, 0.1173396, -2.3557339, 2.3745961

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9674689, upper bound: 0.9674702
time: 5.40 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9674689, upper bound: 0.9703433
time: 6.00 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.3378305, -6.2299337, -9.2880716, -6.2814651, -2.3383899, 2.3471854
1: -6.8579874, -4.3212638, -6.8198347, -4.3332195, -2.1500082, 2.1677403
2: -8.8412151, -6.4588223, -8.8041630, -6.4693136, -2.0555677, 2.0535514
3: -10.2079506, -7.4975433, -10.1439981, -7.5078015, -2.0070405, 1.9792597
4: -5.0471773, -2.4586329, -5.0149965, -2.4794998, -2.3896079, 2.3951197
5: -5.4968190, -2.9152360, -5.4323130, -2.9420798, -2.1342249, 2.1367588
6: -13.7874041, -10.6013517, -13.7059202, -10.6497116, -3.0418181, 3.0329132
7: 3.2235560, 5.0704441, 3.2414141, 5.0245953, -1.6836419, 1.7156725
8: -4.5584836, -1.5123401, -4.4893494, -1.5199366, -2.2689090, 2.2216656
9: -2.3868451, 0.1671113, -2.3652799, 0.1176069, -2.3716526, 2.3958313

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9591894, upper bound: 0.9699947
time: 4.66 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9699947, upper bound: 0.9699948
time: 4.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.88 seconds
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9658417
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
IS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9511470, upper bound: 0.9680033
IS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9617985, upper bound: 0.9680053
IS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9540333, upper bound: 0.9680361
IS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9647101, upper bound: 0.9680371
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9673532, upper bound: 0.9562887
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9673532, upper bound: 0.9670878
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9591746
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9699971
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9674689, upper bound: 0.9674702
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9674689, upper bound: 0.9703433
IS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9591894, upper bound: 0.9699947
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.88
Output dim: 7, lower bound: -0.9699947, upper bound: 0.9699948

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.2152348, -6.2916193, -9.2748222, -6.2840948, -2.2060599, 2.2589943
1: -6.7868690, -4.3616519, -6.8065581, -4.3530550, -2.0330343, 2.0451496
2: -8.7657528, -6.4971304, -8.8043041, -6.4809690, -1.9701910, 1.9817729
3: -10.0937233, -7.5504746, -10.1517401, -7.5329661, -1.8928802, 1.9303453
4: -4.9822993, -2.5226252, -4.9983902, -2.5014248, -2.2873731, 2.2829075
5: -5.3602304, -2.9800000, -5.4418650, -2.9582305, -2.0112801, 2.0509717
6: -13.6672163, -10.7211161, -13.6856966, -10.6719418, -2.8833923, 2.8433037
7: 3.2947311, 5.0180087, 3.2724538, 5.0351205, -1.6309712, 1.6347134
8: -4.4546089, -1.5903029, -4.4984760, -1.5675163, -2.1507878, 2.1700096
9: -2.3205693, 0.0936962, -2.3357840, 0.1208507, -2.3049784, 2.2955565

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9625152
time: 4.03 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9658417
time: 4.25 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.2152348, -6.2916193, -9.3217297, -6.2320967, -2.2513180, 2.2975705
1: -6.7868690, -4.3616519, -6.8484616, -4.3256912, -2.0618610, 2.0919101
2: -8.7657528, -6.4971304, -8.8261204, -6.4678278, -1.9828620, 2.0154839
3: -10.0937233, -7.5504746, -10.1863651, -7.5047603, -1.9216676, 1.9471356
4: -4.9822993, -2.5226252, -5.0356026, -2.4683225, -2.3217940, 2.3155391
5: -5.3602304, -2.9800000, -5.4742250, -2.9243045, -2.0457759, 2.0738027
6: -13.6672163, -10.7211161, -13.7783508, -10.6072121, -2.9344540, 2.9555573
7: 3.2947311, 5.0180087, 3.2422385, 5.0688148, -1.6645999, 1.6604453
8: -4.4546089, -1.5903029, -4.5504475, -1.5320072, -2.1849594, 2.2003009
9: -2.3205693, 0.0936962, -2.3779182, 0.1574709, -2.3377218, 2.3380961

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9513344, upper bound: 0.9680375
time: 4.38 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9620948, upper bound: 0.9680337
time: 4.94 seconds

## BFS IS instance: IS_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -9.2572556, -6.2615566, -9.2708750, -6.2929478, -2.2623291, 2.2915597
1: -6.8211813, -4.3480616, -6.8089199, -4.3424864, -2.1221404, 2.1372666
2: -8.7803631, -6.4984488, -8.7862167, -6.4822426, -1.9941525, 1.9909670
3: -10.1132727, -7.5417647, -10.1176710, -7.5203314, -1.9334896, 1.9159222
4: -5.0095854, -2.5090604, -5.0009394, -2.4965231, -2.3323202, 2.3310318
5: -5.3861189, -2.9699450, -5.4080043, -2.9594979, -2.0687571, 2.0634642
6: -13.7438297, -10.6634789, -13.6908369, -10.6580420, -2.9891224, 2.9302731
7: 3.2850552, 5.0476489, 3.2674198, 5.0223284, -1.6199431, 1.6651282
8: -4.4960327, -1.5851898, -4.4790921, -1.5499868, -2.1896753, 2.1535125
9: -2.3479700, 0.1142001, -2.3523300, 0.1018803, -2.3123908, 2.3303604

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6208

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_A1_A1_B1

### Relational analysis result of IS_A1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9511470, upper bound: 0.9651267
time: 5.03 seconds

## Relational analysis of IS_A1_A2_A1_A1_B2

### Relational analysis result of IS_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9511470, upper bound: 0.9680033
time: 4.77 seconds

## BFS IS instance: IS_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -9.2599201, -6.2573991, -9.2715816, -6.2893944, -2.2680545, 2.2943265
1: -6.8252583, -4.3353786, -6.8098421, -4.3378711, -2.1141782, 2.1523287
2: -8.7861853, -6.4962664, -8.7889652, -6.4817777, -1.9963174, 1.9933054
3: -10.1262455, -7.5395427, -10.1224899, -7.5199466, -1.9447503, 1.9141300
4: -5.0139952, -2.4928324, -5.0016928, -2.4899881, -2.3470025, 2.3355877
5: -5.3879194, -2.9686742, -5.4085646, -2.9583278, -2.0715179, 2.0649738
6: -13.7443914, -10.6579447, -13.6917725, -10.6558247, -2.9957623, 2.9353895
7: 3.2723246, 5.0503516, 3.2624679, 5.0225754, -1.6237283, 1.6731055
8: -4.5014367, -1.5659618, -4.4799805, -1.5428791, -2.2063212, 2.1659203
9: -2.3526464, 0.1287670, -2.3532267, 0.1077271, -2.3234749, 2.3338704

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6208

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_A1_A2_B1

### Relational analysis result of IS_A1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9617985, upper bound: 0.9651276
time: 4.26 seconds

## Relational analysis of IS_A1_A2_A1_A2_B2

### Relational analysis result of IS_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9617985, upper bound: 0.9680053
time: 4.26 seconds

## BFS IS instance: IS_A1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.3021250, -6.2420030, -9.2713566, -6.2870479, -2.2937596, 2.3054233
1: -6.8314962, -4.3439112, -6.8094134, -4.3421268, -2.1330471, 2.1418493
2: -8.8055944, -6.4834404, -8.7867498, -6.4787898, -2.0237055, 2.0168064
3: -10.1500359, -7.5229111, -10.1179934, -7.5149946, -1.9538455, 1.9359088
4: -5.0174913, -2.4969909, -5.0027800, -2.4957838, -2.3415089, 2.3396378
5: -5.4470148, -2.9469385, -5.4092574, -2.9521968, -2.1075664, 2.0816586
6: -13.7649946, -10.6222343, -13.6959829, -10.6576796, -3.0048537, 2.9766531
7: 3.2755575, 5.0641470, 3.2651086, 5.0227590, -1.6282163, 1.6772144
8: -4.5273337, -1.5720720, -4.4803600, -1.5466733, -2.2033491, 2.1756535
9: -2.3612037, 0.1337494, -2.3554804, 0.1021464, -2.3283548, 2.3512082

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A2_A2_A1_B1

### Relational analysis result of IS_A1_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9540333, upper bound: 0.9647084
time: 4.99 seconds

## Relational analysis of IS_A1_A2_A2_A1_B2

### Relational analysis result of IS_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9540333, upper bound: 0.9680361
time: 5.14 seconds

## BFS IS instance: IS_A1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.3047943, -6.2378769, -9.2720642, -6.2835073, -2.3032205, 2.3081915
1: -6.8355756, -4.3312168, -6.8103366, -4.3375130, -2.1250834, 2.1564956
2: -8.8114157, -6.4812684, -8.7895002, -6.4783254, -2.0258765, 2.0191944
3: -10.1630192, -7.5206871, -10.1228132, -7.5146084, -1.9588032, 1.9341333
4: -5.0218921, -2.4807594, -5.0035310, -2.4892473, -2.3561869, 2.3441978
5: -5.4488182, -2.9456692, -5.4098206, -2.9510260, -2.1109052, 2.0831690
6: -13.7655602, -10.6166925, -13.6969194, -10.6554642, -3.0114870, 2.9817753
7: 3.2628284, 5.0668492, 3.2601576, 5.0230055, -1.6320043, 1.6875806
8: -4.5327454, -1.5528498, -4.4812474, -1.5395660, -2.2199998, 2.1880684
9: -2.3658605, 0.1483171, -2.3563776, 0.1079926, -2.3394465, 2.3546886

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6208

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5751

## Relational analysis of IS_A1_A2_A2_A2_B1

### Relational analysis result of IS_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9636408, upper bound: 0.9680238
time: 4.79 seconds

## Relational analysis of IS_A1_A2_A2_A2_B2

### Relational analysis result of IS_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9646989, upper bound: 0.9680268
time: 4.54 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -9.2462196, -6.2932959, -9.2641525, -6.3045225, -2.2278576, 2.2748990
1: -6.8058925, -4.3566723, -6.8078356, -4.3556490, -2.0697103, 2.0841157
2: -8.7918615, -6.4802895, -8.7906303, -6.4852920, -1.9785390, 1.9996452
3: -10.1318331, -7.5341487, -10.1268110, -7.5362959, -1.9218349, 1.9317226
4: -5.0049391, -2.5088773, -5.0015121, -2.5099611, -2.3149829, 2.3119569
5: -5.4054651, -2.9590588, -5.4150019, -2.9714334, -2.0228028, 2.0766144
6: -13.6826763, -10.7088861, -13.6817360, -10.6831608, -2.8914061, 2.9052687
7: 3.2634730, 5.0207815, 3.2693949, 5.0191140, -1.6468725, 1.6379101
8: -4.4768381, -1.5613418, -4.4752831, -1.5637178, -2.1707678, 2.1530919
9: -2.3369951, 0.1052531, -2.3396792, 0.0977904, -2.2987647, 2.3092709

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9562874
time: 4.44 seconds

## Relational analysis of IS_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9562909
time: 4.97 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -9.2470255, -6.2897749, -9.2675686, -6.3003416, -2.2308025, 2.2813263
1: -6.8068833, -4.3520746, -6.8125439, -4.3429565, -2.0928378, 2.0771484
2: -8.7948799, -6.4798236, -8.7983990, -6.4830441, -1.9820595, 2.0034437
3: -10.1366463, -7.5337410, -10.1397810, -7.5336652, -1.9210522, 1.9442291
4: -5.0056829, -2.5021553, -5.0059328, -2.4919915, -2.3254290, 2.3165507
5: -5.4060316, -2.9576426, -5.4168100, -2.9679582, -2.0264382, 2.0795257
6: -13.6836510, -10.7066107, -13.6824179, -10.6770849, -2.8971119, 2.9105606
7: 3.2583513, 5.0210376, 3.2552433, 5.0218387, -1.6550498, 1.6427747
8: -4.4777541, -1.5542264, -4.4807525, -1.5444117, -2.1836429, 2.1659017
9: -2.3379030, 0.1113198, -2.3443851, 0.1140064, -2.3042588, 2.3206391

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9670873
time: 4.73 seconds

## Relational analysis of IS_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9670909
time: 4.73 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -9.2467012, -6.2873955, -9.3090000, -6.2849302, -2.2417278, 2.3082526
1: -6.8063893, -4.3563156, -6.8181424, -4.3515310, -2.0744252, 2.0952971
2: -8.7924032, -6.4768333, -8.8157921, -6.4704332, -2.0045109, 2.0290663
3: -10.1321516, -7.5288019, -10.1637802, -7.5173626, -1.9417939, 1.9673698
4: -5.0067892, -2.5081360, -5.0094786, -2.4979339, -2.3286047, 2.3241277
5: -5.4067159, -2.9517584, -5.4758682, -2.9484081, -2.0409927, 2.1147919
6: -13.6878366, -10.7085295, -13.7030067, -10.6419611, -2.9377270, 2.9205441
7: 3.2611575, 5.0212054, 3.2599177, 5.0356417, -1.6657844, 1.6461897
8: -4.4780817, -1.5580182, -4.5067163, -1.5506315, -2.1928663, 2.1848977
9: -2.3401399, 0.1055160, -2.3528433, 0.1174905, -2.3222051, 2.3253298

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1_B2_B1_B1

### Relational analysis result of IS_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9565004
time: 4.80 seconds

## Relational analysis of IS_A2_A1_B2_B1_B2

### Relational analysis result of IS_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9591746
time: 4.99 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -9.2475061, -6.2838831, -9.3124199, -6.2807841, -2.2446737, 2.3185678
1: -6.8073816, -4.3517175, -6.8228526, -4.3388281, -2.0975547, 2.0883346
2: -8.7954206, -6.4763641, -8.8235588, -6.4681964, -2.0080361, 2.0328684
3: -10.1369629, -7.5283947, -10.1767559, -7.5147305, -1.9410233, 1.9724728
4: -5.0075331, -2.5014160, -5.0138903, -2.4799583, -2.3390141, 2.3287067
5: -5.4072819, -2.9503427, -5.4776807, -2.9449356, -2.0446301, 2.1184056
6: -13.6888123, -10.7062511, -13.7037010, -10.6358843, -2.9434347, 2.9258208
7: 3.2560368, 5.0214620, 3.2457695, 5.0383682, -1.6739619, 1.6510285
8: -4.4790010, -1.5509028, -4.5121965, -1.5313287, -2.2057490, 2.2018068
9: -2.3410478, 0.1115847, -2.3575308, 0.1337081, -2.3277020, 2.3367057

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9673623
time: 5.16 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9699934
time: 5.03 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.2865963, -6.2992158, -2.2895870, 2.3319757
1: -6.8476830, -4.3254142, -6.8183270, -4.3343148, -2.1382780, 2.1621065
2: -8.8160028, -6.4737835, -8.8025284, -6.4797006, -2.0164151, 2.0228474
3: -10.1710596, -7.5164576, -10.1430054, -7.5239162, -1.9653320, 1.9560716
4: -5.0392056, -2.4706733, -5.0094357, -2.4817402, -2.3754787, 2.3810163
5: -5.4359431, -2.9382565, -5.4285116, -2.9640720, -2.0791473, 2.1157770
6: -13.7662449, -10.6425924, -13.6904182, -10.6508083, -3.0253954, 2.9758663
7: 3.2330532, 5.0539637, 3.2483788, 5.0232897, -1.6743007, 1.6920921
8: -4.5272012, -1.5254626, -4.4855080, -1.5299249, -2.2472758, 2.1945667
9: -2.3735833, 0.1474495, -2.3557930, 0.1167958, -2.3533506, 2.3672194

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A2_A2_A1_B1_A1

### Relational analysis result of IS_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9562999, upper bound: 0.9670886
time: 8.91 seconds

## Relational analysis of IS_A2_A2_A1_B1_A2

### Relational analysis result of IS_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9670886, upper bound: 0.9670881
time: 4.65 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2929459, -6.2494793, -9.3314667, -6.2796702, -2.3088813, 2.3371899
1: -6.8476830, -4.3254142, -6.8286362, -4.3301487, -2.1423926, 2.1729829
2: -8.8160028, -6.4737835, -8.8276863, -6.4648428, -2.0311341, 2.0475092
3: -10.1710596, -7.5164576, -10.1799879, -7.5049729, -1.9840455, 1.9936478
4: -5.0392056, -2.4706733, -5.0174179, -2.4696643, -2.3834357, 2.3888454
5: -5.4359431, -2.9382565, -5.4894176, -2.9410372, -2.1022029, 2.1323802
6: -13.7662449, -10.6425924, -13.7115936, -10.6095715, -3.0285711, 2.9963155
7: 3.2330532, 5.0539637, 3.2388978, 5.0398245, -1.6908741, 1.7014501
8: -4.5272012, -1.5254626, -4.5169911, -1.5168543, -2.2569723, 2.2262664
9: -2.3735833, 0.1474495, -2.3689320, 0.1365037, -2.3731570, 2.3805060

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A2_A2_A1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9562999, upper bound: 0.9699697
time: 6.72 seconds

## Relational analysis of IS_A2_A2_A1_B2_A2

### Relational analysis result of IS_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9670886, upper bound: 0.9699693
time: 4.99 seconds

## BFS IS instance: IS_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.3342609, -6.2340889, -9.2872639, -6.2849984, -2.3260984, 2.3354273
1: -6.8532782, -4.3339796, -6.8188486, -4.3378363, -2.1571469, 2.1620297
2: -8.8333998, -6.4610910, -8.8011541, -6.4697828, -2.0516701, 2.0505188
3: -10.1949930, -7.5002508, -10.1391859, -7.5082245, -1.9932215, 1.9806404
4: -5.0427060, -2.4766455, -5.0142369, -2.4862366, -2.3722296, 2.3766859
5: -5.4949284, -2.9187679, -5.4317455, -2.9434941, -2.1303358, 2.1334755
6: -13.7867613, -10.6074238, -13.7049789, -10.6519880, -3.0346918, 3.0252142
7: 3.2377038, 5.0676923, 3.2465463, 5.0243468, -1.6692989, 1.7036109
8: -4.5529752, -1.5317149, -4.4884481, -1.5270452, -2.2489138, 2.2009323
9: -2.3820417, 0.1509035, -2.3643758, 0.1115420, -2.3601270, 2.3780236

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9591724, upper bound: 0.9670887
time: 5.39 seconds

## Relational analysis of IS_A2_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9591727, upper bound: 0.9671395
time: 4.92 seconds

## BFS IS instance: IS_A2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.3377028, -6.2299562, -9.2880716, -6.2814651, -2.3364248, 2.3385680
1: -6.8579721, -4.3212757, -6.8198347, -4.3332195, -2.1499906, 2.1758916
2: -8.8411264, -6.4588356, -8.8041630, -6.4693136, -2.0555015, 2.0535347
3: -10.2079468, -7.4976130, -10.1439981, -7.5078015, -1.9983330, 1.9788568
4: -5.0471401, -2.4586637, -5.0149965, -2.4794998, -2.3872948, 2.3828247
5: -5.4967847, -2.9152851, -5.4323130, -2.9420798, -2.1341028, 2.1366742
6: -13.7873249, -10.6013565, -13.7059202, -10.6497116, -3.0414310, 3.0309339
7: 3.2235627, 5.0704150, 3.2414141, 5.0245953, -1.6741180, 1.7141817
8: -4.5584469, -1.5124121, -4.4893494, -1.5199366, -2.2658095, 2.2137957
9: -2.3867660, 0.1670907, -2.3652799, 0.1176069, -2.3715692, 2.3834395

Time for backsubstitution: 12.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9680349, upper bound: 0.9647109
time: 4.59 seconds

## Relational analysis of IS_A2_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9680352, upper bound: 0.9672124
time: 4.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.58 seconds
IS_A1_A1_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9625152
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9658417
IS_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9513344, upper bound: 0.9680375
IS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9620948, upper bound: 0.9680337
IS_A1_A2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9511470, upper bound: 0.9651267
IS_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9511470, upper bound: 0.9680033
IS_A1_A2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9617985, upper bound: 0.9651276
IS_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9617985, upper bound: 0.9680053
IS_A1_A2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9540333, upper bound: 0.9647084
IS_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9540333, upper bound: 0.9680361
IS_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9636408, upper bound: 0.9680238
IS_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9646989, upper bound: 0.9680268
IS_A2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9562874
IS_A2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9562909
IS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9670873
IS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9644694, upper bound: 0.9670909
IS_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9565004
IS_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9591746
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9673623
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9673639, upper bound: 0.9699934
IS_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9562999, upper bound: 0.9670886
IS_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9670886, upper bound: 0.9670881
IS_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9562999, upper bound: 0.9699697
IS_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9670886, upper bound: 0.9699693
IS_A2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9591724, upper bound: 0.9670887
IS_A2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9591727, upper bound: 0.9671395
IS_A2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9680349, upper bound: 0.9647109
IS_A2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 22.58
Output dim: 7, lower bound: -0.9680352, upper bound: 0.9672124

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -9.2152348, -6.2916193, -9.2908669, -6.2820559, -2.2105856, 2.2695491
1: -6.7868690, -4.3616519, -6.8161855, -4.3487291, -2.0374875, 2.0525179
2: -8.7657528, -6.4971304, -8.8189554, -6.4719419, -1.9766831, 1.9963078
3: -10.0937233, -7.5504746, -10.1729650, -7.5255885, -1.8992493, 1.9413968
4: -4.9822993, -2.5226252, -5.0099001, -2.4916728, -2.2944417, 2.2951894
5: -5.3602304, -2.9800000, -5.4643111, -2.9493253, -2.0178480, 2.0563798
6: -13.6672163, -10.7211161, -13.6946964, -10.6661882, -2.8856831, 2.8528786
7: 3.2947311, 5.0180087, 3.2535410, 5.0367002, -1.6325772, 1.6547204
8: -4.4546089, -1.5903029, -4.5066185, -1.5478115, -2.1693134, 2.1718669
9: -2.3205693, 0.0936962, -2.3447146, 0.1304823, -2.3162746, 2.3039246

Time for backsubstitution: 12.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 551
type: A, layer: 1, pos: 551

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 874

## Relational analysis of IS_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9620951, upper bound: 0.9546532
time: 5.08 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9620951, upper bound: 0.9654187
time: 4.66 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2124863, -6.2957644, -9.3210144, -6.2356138, -2.2399054, 2.2860234
1: -6.7829547, -4.3743172, -6.8475432, -4.3303061, -2.0699010, 2.0864286
2: -8.7598333, -6.4992275, -8.8233795, -6.4682932, -1.9806571, 2.0119946
3: -10.0808697, -7.5528107, -10.1815529, -7.5051441, -1.9158559, 1.9346225
4: -4.9778833, -2.5387745, -5.0348520, -2.4748561, -2.3186870, 2.2987819
5: -5.3584867, -2.9813144, -5.4736490, -2.9254782, -2.0429735, 2.0715280
6: -13.6664877, -10.7266464, -13.7774410, -10.6094275, -2.9276066, 2.9484372
7: 3.3074217, 5.0152597, 3.2471852, 5.0685673, -1.6512692, 1.6524187
8: -4.4492283, -1.6095843, -4.5495539, -1.5391197, -2.1653762, 2.1799662
9: -2.3158672, 0.0791600, -2.3770049, 0.1516303, -2.3205957, 2.3223109

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9513344, upper bound: 0.9647106
time: 4.70 seconds

## Relational analysis of IS_A1_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9513344, upper bound: 0.9680375
time: 4.72 seconds

## BFS IS instance: IS_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.2151194, -6.2916384, -9.3217297, -6.2320967, -2.2493668, 2.2887874
1: -6.7868528, -4.3616638, -6.8484616, -4.3256912, -2.0618434, 2.1093824
2: -8.7656698, -6.4971428, -8.8261204, -6.4678278, -1.9828143, 2.0154653
3: -10.0937176, -7.5505471, -10.1863651, -7.5047603, -1.9275703, 1.9449002
4: -4.9822617, -2.5226543, -5.0356026, -2.4683225, -2.3217597, 2.3032448
5: -5.3601990, -2.9800563, -5.4742250, -2.9243045, -2.0457144, 2.0734003
6: -13.6671410, -10.7211199, -13.7783508, -10.6072121, -2.9340820, 2.9541364
7: 3.2947364, 5.0179801, 3.2422385, 5.0688148, -1.6550517, 1.6604167
8: -4.4545660, -1.5903592, -4.5504475, -1.5320072, -2.1818566, 2.1924241
9: -2.3204927, 0.0936768, -2.3779182, 0.1574709, -2.3350320, 2.3257637

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5751

## Relational analysis of IS_A1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9610270, upper bound: 0.9680263
time: 4.45 seconds

## Relational analysis of IS_A1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9620829, upper bound: 0.9680224
time: 4.98 seconds

## BFS IS instance: IS_A1_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2572556, -6.2615566, -9.3147545, -6.2852635, -2.2697935, 2.2954679
1: -6.8211813, -4.3480616, -6.8182249, -4.3390522, -2.1255302, 2.1469464
2: -8.7803631, -6.4984488, -8.8102932, -6.4743009, -1.9992704, 2.0057600
3: -10.1132727, -7.5417647, -10.1539488, -7.5121627, -1.9405289, 1.9359320
4: -5.0095854, -2.5090604, -5.0051918, -2.4859338, -2.3353601, 2.3338084
5: -5.3861189, -2.9699450, -5.4663844, -2.9511633, -2.0771284, 2.0779278
6: -13.7438297, -10.6634789, -13.7016668, -10.6175423, -2.9915714, 2.9400945
7: 3.2850552, 5.0476489, 3.2625895, 5.0379963, -1.6354685, 1.6697640
8: -4.4960327, -1.5851898, -4.5080113, -1.5435953, -2.1915245, 2.1801786
9: -2.3479700, 0.1142001, -2.3591046, 0.1210082, -2.3297796, 2.3362465

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6208

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_A2_A1_A1_B2_B1

### Relational analysis result of IS_A1_A2_A1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9511470, upper bound: 0.9646756
time: 4.98 seconds

## Relational analysis of IS_A1_A2_A1_A1_B2_B2

### Relational analysis result of IS_A1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9511488, upper bound: 0.9680033
time: 5.12 seconds

## BFS IS instance: IS_A1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2599201, -6.2573991, -9.3154650, -6.2817225, -2.2755060, 2.2982383
1: -6.8252583, -4.3353786, -6.8191471, -4.3344345, -2.1175685, 2.1600559
2: -8.7861853, -6.4962664, -8.8130436, -6.4738412, -2.0014300, 2.0119870
3: -10.1262455, -7.5395427, -10.1587706, -7.5117803, -1.9476089, 1.9462155
4: -5.0139952, -2.4928324, -5.0059385, -2.4793954, -2.3500466, 2.3383584
5: -5.3879194, -2.9686742, -5.4669456, -2.9499936, -2.0798879, 2.0798399
6: -13.7443914, -10.6579447, -13.7025928, -10.6153221, -2.9982128, 2.9452019
7: 3.2723246, 5.0503516, 3.2576375, 5.0382423, -1.6392536, 1.6777415
8: -4.5014367, -1.5659618, -4.5089030, -1.5364871, -2.2081747, 2.1925917
9: -2.3526464, 0.1287670, -2.3599982, 0.1268563, -2.3408666, 2.3397532

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 6208

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5751

## Relational analysis of IS_A1_A2_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9607230, upper bound: 0.9679953
time: 4.90 seconds

## Relational analysis of IS_A1_A2_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9617873, upper bound: 0.9679930
time: 4.52 seconds

## BFS IS instance: IS_A1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.3021250, -6.2420030, -9.2863617, -6.2850103, -2.2959082, 2.3149984
1: -6.8314962, -4.3439112, -6.8181705, -4.3378611, -2.1372957, 2.1481595
2: -8.8055944, -6.4834404, -8.7989998, -6.4697905, -2.0292025, 2.0290608
3: -10.1500359, -7.5229111, -10.1391735, -7.5086603, -1.9543743, 1.9583843
4: -5.0174913, -2.4969909, -5.0142064, -2.4881494, -2.3449922, 2.3444908
5: -5.4470148, -2.9469385, -5.4316902, -2.9459534, -2.1000433, 2.0921483
6: -13.7649946, -10.6222343, -13.7049484, -10.6525583, -3.0028572, 2.9861941
7: 3.2755575, 5.0641470, 3.2482190, 5.0243254, -1.6298056, 1.6892831
8: -4.5273337, -1.5720720, -4.4883924, -1.5274391, -2.2095890, 2.1828930
9: -2.3612037, 0.1337494, -2.3643131, 0.1097572, -2.3371816, 2.3587220

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5751
type: A, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6235
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_A2_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9540117, upper bound: 0.9651268
time: 5.41 seconds

## Relational analysis of IS_A1_A2_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9540120, upper bound: 0.9651264
time: 4.63 seconds

## BFS IS instance: IS_A1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.3047943, -6.2378769, -9.2713470, -6.2837701, -2.3024092, 2.3068299
1: -6.8355756, -4.3312168, -6.8094382, -4.3377962, -2.1246982, 2.1554809
2: -8.8114157, -6.4812684, -8.7878666, -6.4787931, -2.0252113, 2.0172265
3: -10.1630192, -7.5206871, -10.1208878, -7.5152550, -1.9581549, 1.9322224
4: -5.0218921, -2.4807594, -5.0029163, -2.4909747, -2.3545184, 2.3437243
5: -5.4488182, -2.9456692, -5.4090524, -2.9523396, -2.1082659, 2.0812488
6: -13.7655602, -10.6166925, -13.6962805, -10.6560097, -3.0100207, 2.9803782
7: 3.2628284, 5.0668492, 3.2605948, 5.0223508, -1.6310172, 1.6871295
8: -4.5327454, -1.5528498, -4.4800143, -1.5399928, -2.2195458, 2.1867323
9: -2.3658605, 0.1483171, -2.3555746, 0.1077027, -2.3380146, 2.3530183

Time for backsubstitution: 12.64 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.6687421798706055
rel_dist={7: [-0.9703767544746134, 0.9703745834961706]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2427.67 seconds
