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
execution time: IAR + LP analysis = 13.27 + 33.05 = 46.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.4649108, upper bound: 1.4649096


# Binary Search by BASE starts (time budget: 3553.68 seconds, max iter: 100)

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
Binary search time: 143.48 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01171875


# Individual Split (IS_dual_ind) starts
Time budget: 3410.20 seconds

## Binary search (step 0) starts
Candidate k: 8, corresponding eps: 0.0312500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621139, upper bound: 1.2674239
time: 4.24 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674215, upper bound: 1.2674238
time: 4.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 8.55
Output dim: 7, lower bound: -1.2621139, upper bound: 1.2674239
IS_A2, status: Status.UNKNOWN, split count: 1, time: 8.55
Output dim: 7, lower bound: -1.2674215, upper bound: 1.2674238

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.2475243, -6.2838678, -9.2831097, -6.2817378, -2.6666651, 2.7157860
1: -6.8073869, -4.3517122, -6.8183289, -4.3354921, -2.3786583, 2.3754511
2: -8.7954388, -6.4763551, -8.8031130, -6.4701843, -2.2875023, 2.2982733
3: -10.1369781, -7.5283747, -10.1431608, -7.5103340, -2.3082938, 2.2949340
4: -5.0075459, -2.5014083, -5.0140982, -2.4821897, -2.5253563, 2.5126898
5: -5.4072948, -2.9503248, -5.4292421, -2.9430604, -2.4079666, 2.4197335
6: -13.6888247, -10.7062445, -13.7038717, -10.6566191, -3.0322056, 2.9976273
7: 3.2560186, 5.0214634, 3.2432132, 5.0242224, -1.7682037, 1.7782502
8: -4.4790077, -1.5508766, -4.4881186, -1.5237064, -2.5896964, 2.5729361
9: -2.3410625, 0.1115901, -2.3622925, 0.1168859, -2.4579484, 2.4738827

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621097, upper bound: 1.2615699
time: 4.35 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621098, upper bound: 1.2674176
time: 4.34 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.2944317, -6.2317004, -9.2880936, -6.2814474, -2.7423110, 2.7708218
1: -6.8492031, -4.3243413, -6.8198419, -4.3332100, -2.4286737, 2.4636073
2: -8.8176489, -6.4633946, -8.8041744, -6.4693046, -2.3230438, 2.3254166
3: -10.1720686, -7.5003147, -10.1440058, -7.5077753, -2.3478768, 2.3243074
4: -5.0447531, -2.4684689, -5.0150080, -2.4794893, -2.5652637, 2.5465391
5: -5.4397259, -2.9162605, -5.4323239, -2.9420631, -2.4674039, 2.4594190
6: -13.7818432, -10.6415081, -13.7059317, -10.6496916, -3.1321516, 3.0644236
7: 3.2260804, 5.0552716, 3.2413988, 5.0245962, -1.7985158, 1.8138728
8: -4.5310378, -1.5154338, -4.4893570, -1.5199089, -2.6488118, 2.6090813
9: -2.3830755, 0.1482754, -2.3652978, 0.1176099, -2.5006852, 2.5135732

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674174, upper bound: 1.2615716
time: 4.37 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674174, upper bound: 1.2674193
time: 4.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.22
Output dim: 7, lower bound: -1.2621097, upper bound: 1.2615699
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.22
Output dim: 7, lower bound: -1.2621098, upper bound: 1.2674176
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.22
Output dim: 7, lower bound: -1.2674174, upper bound: 1.2615716
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.22
Output dim: 7, lower bound: -1.2674174, upper bound: 1.2674193

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.2475243, -6.2838678, -9.2816353, -6.2995005, -2.6489134, 2.7138281
1: -6.8073869, -4.3517122, -6.8168201, -4.3365870, -2.3775816, 2.3739090
2: -8.7954388, -6.4763551, -8.8014803, -6.4805808, -2.2731409, 2.2910175
3: -10.1369781, -7.5283747, -10.1421719, -7.5264635, -2.2908349, 2.2900934
4: -5.0075459, -2.5014083, -5.0085325, -2.4844325, -2.5231135, 2.5071242
5: -5.4072948, -2.9503248, -5.4254379, -2.9650645, -2.3859687, 2.4155273
6: -13.6888247, -10.7062445, -13.6883631, -10.6577196, -3.0311050, 2.9821186
7: 3.2560186, 5.0214634, 3.2501817, 5.0229197, -1.7669010, 1.7712817
8: -4.4790077, -1.5508766, -4.4842796, -1.5337062, -2.5764985, 2.5654752
9: -2.3410625, 0.1115901, -2.3527982, 0.1160749, -2.4571376, 2.4643884

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562617, upper bound: 1.2615698
time: 4.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562616, upper bound: 1.2615693
time: 4.52 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.2475224, -6.2838745, -9.3264999, -6.2799554, -2.6709065, 2.7582281
1: -6.8073854, -4.3517132, -6.8271284, -4.3324289, -2.3818140, 2.3846476
2: -8.7954378, -6.4763565, -8.8266335, -6.4657254, -2.3043780, 2.3156805
3: -10.1369781, -7.5283823, -10.1791515, -7.5075221, -2.3181040, 2.3276656
4: -5.0075455, -2.5014079, -5.0165081, -2.4723678, -2.5351777, 2.5151002
5: -5.4072928, -2.9503307, -5.4863348, -2.9420314, -2.4117417, 2.4702268
6: -13.6888237, -10.7062445, -13.7095671, -10.6164894, -3.0723343, 3.0033226
7: 3.2560196, 5.0214643, 3.2407026, 5.0394526, -1.7834330, 1.7807617
8: -4.4790058, -1.5508795, -4.5157537, -1.5206332, -2.6033616, 2.5970778
9: -2.3410602, 0.1115904, -2.3659365, 0.1357831, -2.4768434, 2.4775269

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2634924
time: 6.42 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2674115
time: 4.50 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.2944317, -6.2317004, -9.2866173, -6.2992134, -2.7245884, 2.7688689
1: -6.8492031, -4.3243413, -6.8183327, -4.3343067, -2.4275851, 2.4620643
2: -8.8176489, -6.4633946, -8.8025370, -6.4796963, -2.3087058, 2.3181586
3: -10.1720686, -7.5003147, -10.1430130, -7.5239034, -2.3304143, 2.3194680
4: -5.0447531, -2.4684689, -5.0094433, -2.4817321, -2.5630209, 2.5409744
5: -5.4397259, -2.9162605, -5.4285192, -2.9640665, -2.4454064, 2.4553108
6: -13.7818432, -10.6415081, -13.6904249, -10.6507912, -3.1310520, 3.0489168
7: 3.2260804, 5.0552716, 3.2483668, 5.0232906, -1.7972102, 1.8069048
8: -4.5310378, -1.5154338, -4.4855089, -1.5299053, -2.6356120, 2.6016197
9: -2.3830755, 0.1482754, -2.3558052, 0.1167988, -2.4998741, 2.5040805

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2615694, upper bound: 1.2615716
time: 4.43 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2615692, upper bound: 1.2615693
time: 4.24 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.2944317, -6.2317095, -9.3314886, -6.2796688, -2.7465367, 2.7740822
1: -6.8492031, -4.3243408, -6.8286424, -4.3301387, -2.4318066, 2.4728098
2: -8.8176498, -6.4633980, -8.8276949, -6.4648376, -2.3399668, 2.3428154
3: -10.1720676, -7.5003228, -10.1799955, -7.5049610, -2.3576722, 2.3570409
4: -5.0447507, -2.4684694, -5.0174251, -2.4696565, -2.5750942, 2.5489557
5: -5.4397249, -2.9162664, -5.4894247, -2.9410322, -2.4711809, 2.4931302
6: -13.7818394, -10.6415091, -13.7115984, -10.6095543, -3.1722851, 3.0700893
7: 3.2260838, 5.0552702, 3.2388849, 5.0398254, -1.8137417, 1.8163853
8: -4.5310345, -1.5154386, -4.5169950, -1.5168343, -2.6603608, 2.6332369
9: -2.3830724, 0.1482766, -2.3689446, 0.1365077, -2.5195801, 2.5172212

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576186, upper bound: 1.2634926
time: 4.52 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2674110, upper bound: 1.2674132
time: 4.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2562617, upper bound: 1.2615698
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2562616, upper bound: 1.2615693
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2634924
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2674115
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2615694, upper bound: 1.2615716
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2615692, upper bound: 1.2615693
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2576186, upper bound: 1.2634926
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.78
Output dim: 7, lower bound: -1.2674110, upper bound: 1.2674132

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2460556, -6.3016310, -9.2816353, -6.2995005, -2.6469722, 2.6960764
1: -6.8058786, -4.3528051, -6.8168201, -4.3365870, -2.3760443, 2.3728592
2: -8.7937965, -6.4867706, -8.8014803, -6.4805808, -2.2658834, 2.2767096
3: -10.1360102, -7.5444994, -10.1421719, -7.5264635, -2.2859707, 2.2726383
4: -5.0019784, -2.5036502, -5.0085325, -2.4844325, -2.5175459, 2.5048823
5: -5.4034948, -2.9723291, -5.4254379, -2.9650645, -2.3818860, 2.3935390
6: -13.6732826, -10.7073383, -13.6883631, -10.6577196, -3.0155630, 2.9810247
7: 3.2629938, 5.0201759, 3.2501817, 5.0229197, -1.7599258, 1.7699943
8: -4.4752259, -1.5608859, -4.4842796, -1.5337062, -2.5690398, 2.5522771
9: -2.3315849, 0.1107922, -2.3527982, 0.1160749, -2.4476600, 2.4635904

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562617, upper bound: 1.2562622
time: 4.45 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562617, upper bound: 1.2615698
time: 4.55 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.2908850, -6.2820530, -9.2816353, -6.2995005, -2.6916151, 2.7154012
1: -6.8161888, -4.3487253, -6.8168201, -4.3365870, -2.3867407, 2.3769457
2: -8.8189726, -6.4719353, -8.8014803, -6.4805808, -2.2905779, 2.2914824
3: -10.1729803, -7.5255833, -10.1421719, -7.5264635, -2.3234935, 2.2913322
4: -5.0099111, -2.4916658, -5.0085325, -2.4844325, -2.5254786, 2.5168667
5: -5.4643207, -2.9493170, -5.4254379, -2.9650645, -2.4400840, 2.4165969
6: -13.6947041, -10.6661835, -13.6883631, -10.6577196, -3.0369844, 3.0221796
7: 3.2535262, 5.0367031, 3.2501817, 5.0229197, -1.7693934, 1.7865214
8: -4.5066218, -1.5477891, -4.4842796, -1.5337062, -2.6005654, 2.5656435
9: -2.3447247, 0.1304879, -2.3527982, 0.1160749, -2.4607997, 2.4832861

Time for backsubstitution: 13.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562616, upper bound: 1.2562617
time: 4.72 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562639, upper bound: 1.2615697
time: 4.87 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.2152367, -6.2916131, -9.3183136, -6.2809858, -2.6343174, 2.7346251
1: -6.7868690, -4.3616514, -6.8223562, -4.3346119, -2.3585658, 2.3703475
2: -8.7657547, -6.4971280, -8.8191729, -6.4702406, -2.2706075, 2.2884183
3: -10.0937204, -7.5504661, -10.1683817, -7.5110531, -2.2698648, 2.2941751
4: -4.9823008, -2.5226254, -5.0106764, -2.4772446, -2.5050561, 2.4880509
5: -5.3602324, -2.9799948, -5.4749541, -2.9465106, -2.3594341, 2.4152663
6: -13.6672192, -10.7211161, -13.7049942, -10.6193810, -3.0478382, 2.9838781
7: 3.2947292, 5.0180106, 3.2502518, 5.0386515, -1.7439222, 1.7677588
8: -4.4546103, -1.5902982, -4.5116825, -1.5306273, -2.5640240, 2.5551934
9: -2.3205700, 0.0936956, -2.3614280, 0.1308662, -2.4514360, 2.4551237

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2581826
time: 4.41 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2634919
time: 6.73 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838755, -9.3264999, -6.2799554, -2.6660466, 2.7560389
1: -6.8073831, -4.3517170, -6.8271284, -4.3324289, -2.3802385, 2.3875029
2: -8.7954197, -6.4763618, -8.8266335, -6.4657254, -2.2986436, 2.3143797
3: -10.1369629, -7.5283861, -10.1791515, -7.5075221, -2.3060782, 2.3276427
4: -5.0075345, -2.5014138, -5.0165081, -2.4723678, -2.5351667, 2.5150943
5: -5.4072838, -2.9503393, -5.4863348, -2.9420314, -2.3844943, 2.4553192
6: -13.6888151, -10.7062531, -13.7095671, -10.6164894, -3.0723257, 3.0033140
7: 3.2560353, 5.0214629, 3.2407026, 5.0394526, -1.7834172, 1.7807603
8: -4.4790020, -1.5509005, -4.5157537, -1.5206332, -2.6009316, 2.5811129
9: -2.3410516, 0.1115841, -2.3659365, 0.1357831, -2.4768348, 2.4775205

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2621040
time: 4.71 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621033, upper bound: 1.2674117
time: 4.67 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.2929640, -6.2494798, -9.2866173, -6.2992134, -2.7226291, 2.7510629
1: -6.8476877, -4.3254099, -6.8183327, -4.3343067, -2.4260383, 2.4609861
2: -8.8160191, -6.4737797, -8.8025370, -6.4796963, -2.3014879, 2.3038280
3: -10.1710749, -7.5164533, -10.1430130, -7.5239034, -2.3255515, 2.3020205
4: -5.0392170, -2.4706659, -5.0094433, -2.4817321, -2.5574849, 2.5387774
5: -5.4359522, -2.9382517, -5.4285192, -2.9640665, -2.4412994, 2.4333348
6: -13.7662506, -10.6425905, -13.6904249, -10.6507912, -3.1154594, 3.0478344
7: 3.2330379, 5.0539656, 3.2483668, 5.0232906, -1.7902527, 1.8055987
8: -4.5272059, -1.5254412, -4.4855089, -1.5299053, -2.6280341, 2.5884161
9: -2.3735921, 0.1474540, -2.3558052, 0.1167988, -2.4903908, 2.5032592

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2517585
time: 3.93 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2615632, upper bound: 1.2615653
time: 4.40 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.3378448, -6.2299337, -9.2866173, -6.2992134, -2.7602406, 2.7704279
1: -6.8579912, -4.3212600, -6.8183327, -4.3343067, -2.4367375, 2.4649796
2: -8.8412304, -6.4588165, -8.8025370, -6.4796963, -2.3262830, 2.3185053
3: -10.2079659, -7.4975381, -10.1430130, -7.5239034, -2.3601241, 2.3207316
4: -5.0471888, -2.4586282, -5.0094433, -2.4817321, -2.5654566, 2.5508151
5: -5.4968271, -2.9152298, -5.4285192, -2.9640665, -2.4833050, 2.4564075
6: -13.7874107, -10.6013489, -13.6904249, -10.6507912, -3.1366196, 3.0890760
7: 3.2235408, 5.0704460, 3.2483668, 5.0232906, -1.7997499, 1.8220792
8: -4.5584888, -1.5123181, -4.4855089, -1.5299053, -2.6424465, 2.6017952
9: -2.3868537, 0.1671182, -2.3558052, 0.1167988, -2.5036526, 2.5229235

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576310, upper bound: 1.2517578
time: 4.06 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2615631, upper bound: 1.2615629
time: 4.52 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396107, -9.3233061, -6.2807007, -2.7094650, 2.7498403
1: -6.8267608, -4.3343019, -6.8238788, -4.3323183, -2.4068389, 2.4583457
2: -8.7878790, -6.4858303, -8.8202248, -6.4693532, -2.3062663, 2.3142970
3: -10.1272211, -7.5233607, -10.1692276, -7.5084038, -2.3077719, 2.3231173
4: -5.0195069, -2.4906087, -5.0115948, -2.4745324, -2.5449746, 2.5209861
5: -5.3917093, -2.9466329, -5.4780421, -2.9455128, -2.4178944, 2.4372797
6: -13.7600489, -10.6568575, -13.7070255, -10.6124458, -3.1476030, 3.0501680
7: 3.2653728, 5.0516901, 3.2484221, 5.0390220, -1.7736492, 1.8032680
8: -4.5052967, -1.5559220, -4.5129275, -1.5268250, -2.6128540, 2.5907702
9: -2.3622141, 0.1295742, -2.3644352, 0.1315889, -2.4938030, 2.4940095

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576185, upper bound: 1.2581806
time: 4.12 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576188, upper bound: 1.2581805
time: 4.87 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317109, -9.3314886, -6.2796688, -2.7416763, 2.7718706
1: -6.8491974, -4.3243456, -6.8286424, -4.3301387, -2.4301319, 2.4755993
2: -8.8176308, -6.4634013, -8.8276949, -6.4648376, -2.3342199, 2.3415060
3: -10.1720524, -7.5003276, -10.1799955, -7.5049610, -2.3456450, 2.3570352
4: -5.0447407, -2.4684758, -5.0174251, -2.4696565, -2.5750842, 2.5489492
5: -5.4397163, -2.9162726, -5.4894247, -2.9410322, -2.4439163, 2.4782233
6: -13.7818317, -10.6415138, -13.7115984, -10.6095543, -3.1722775, 3.0700846
7: 3.2260985, 5.0552688, 3.2388849, 5.0398254, -1.8137269, 1.8163838
8: -4.5310316, -1.5154614, -4.5169950, -1.5168343, -2.6498213, 2.6174278
9: -2.3830624, 0.1482716, -2.3689446, 0.1365077, -2.5195701, 2.5172162

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634925, upper bound: 1.2576156
time: 4.47 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2634925, upper bound: 1.2674120
time: 4.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.46 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2562617, upper bound: 1.2562622
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2562617, upper bound: 1.2615698
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2562616, upper bound: 1.2562617
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2562639, upper bound: 1.2615697
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2523068, upper bound: 1.2581826
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2523067, upper bound: 1.2634919
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2621034, upper bound: 1.2621040
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2621033, upper bound: 1.2674117
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2517585
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2615632, upper bound: 1.2615653
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2576310, upper bound: 1.2517578
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2615631, upper bound: 1.2615629
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2576185, upper bound: 1.2581806
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2576188, upper bound: 1.2581805
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2634925, upper bound: 1.2576156
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.46
Output dim: 7, lower bound: -1.2634925, upper bound: 1.2674120

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2460556, -6.3016310, -9.2460556, -6.3016310, -2.6377330, 2.6377332
1: -6.8058786, -4.3528051, -6.8058786, -4.3528051, -2.3371882, 2.3371882
2: -8.7937965, -6.4867706, -8.7937965, -6.4867706, -2.2526875, 2.2526872
3: -10.1360102, -7.5444994, -10.1360102, -7.5444994, -2.2659454, 2.2659452
4: -5.0019784, -2.5036502, -5.0019784, -2.5036502, -2.4983282, 2.4983282
5: -5.4034948, -2.9723291, -5.4034948, -2.9723291, -2.3602772, 2.3602769
6: -13.6732826, -10.7073383, -13.6732826, -10.7073383, -2.9659443, 2.9659443
7: 3.2629938, 5.0201759, 3.2629938, 5.0201759, -1.7571821, 1.7571821
8: -4.4752259, -1.5608859, -4.4752259, -1.5608859, -2.5415497, 2.5415497
9: -2.3315849, 0.1107922, -2.3315849, 0.1107922, -2.4423771, 2.4423771

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2523218
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562553, upper bound: 1.2562579
time: 4.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2460556, -6.3016310, -9.2929640, -6.2494798, -2.6898665, 2.7019153
1: -6.8058786, -4.3528051, -6.8476877, -4.3254099, -2.3662019, 2.3850231
2: -8.7937965, -6.4867706, -8.8160191, -6.4737797, -2.2652674, 2.2884579
3: -10.1360102, -7.5444994, -10.1710749, -7.5164533, -2.2947810, 2.3027146
4: -5.0019784, -2.5036502, -5.0392170, -2.4706659, -2.5313125, 2.5355668
5: -5.4034948, -2.9723291, -5.4359522, -2.9382517, -2.3968439, 2.3932767
6: -13.6732826, -10.7073383, -13.7662506, -10.6425905, -3.0306921, 3.0589123
7: 3.2629938, 5.0201759, 3.2330379, 5.0539656, -1.7909718, 1.7871380
8: -4.4752259, -1.5608859, -4.5272059, -1.5254412, -2.5776310, 2.5966997
9: -2.3315849, 0.1107922, -2.3735921, 0.1474540, -2.4790390, 2.4843843

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2576310
time: 5.38 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562553, upper bound: 1.2615656
time: 4.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2908850, -6.2820530, -9.2460556, -6.3016310, -2.6823769, 2.6570971
1: -6.8161888, -4.3487253, -6.8058786, -4.3528051, -2.3478842, 2.3412757
2: -8.8189726, -6.4719353, -8.7937965, -6.4867706, -2.2773814, 2.2674415
3: -10.1729803, -7.5255833, -10.1360102, -7.5444994, -2.3034682, 2.2846384
4: -5.0099111, -2.4916658, -5.0019784, -2.5036502, -2.5062609, 2.5103126
5: -5.4643207, -2.9493170, -5.4034948, -2.9723291, -2.4171429, 2.3833349
6: -13.6947041, -10.6661835, -13.6732826, -10.7073383, -2.9873657, 3.0070992
7: 3.2535262, 5.0367031, 3.2629938, 5.0201759, -1.7666497, 1.7737093
8: -4.5066218, -1.5477891, -4.4752259, -1.5608859, -2.5730753, 2.5549321
9: -2.3447247, 0.1304879, -2.3315849, 0.1107922, -2.4555168, 2.4620728

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523064, upper bound: 1.2523198
time: 4.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621030, upper bound: 1.2562580
time: 4.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2908850, -6.2820530, -9.2929640, -6.2494798, -2.6951437, 2.7212400
1: -6.8161888, -4.3487253, -6.8476877, -4.3254099, -2.3768983, 2.3891094
2: -8.8189726, -6.4719353, -8.8160191, -6.4737797, -2.2899613, 2.3032310
3: -10.1729803, -7.5255833, -10.1710749, -7.5164533, -2.3323038, 2.3214085
4: -5.0099111, -2.4916658, -5.0392170, -2.4706659, -2.5392451, 2.5475512
5: -5.4643207, -2.9493170, -5.4359522, -2.9382517, -2.4347019, 2.4163346
6: -13.6947041, -10.6661835, -13.7662506, -10.6425905, -3.0521135, 3.1000671
7: 3.2535262, 5.0367031, 3.2330379, 5.0539656, -1.8004394, 1.8036652
8: -4.5066218, -1.5477891, -4.5272059, -1.5254412, -2.6091566, 2.6100664
9: -2.3447247, 0.1304879, -2.3735921, 0.1474540, -2.4921787, 2.5040801

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523064, upper bound: 1.2576320
time: 4.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2621028, upper bound: 1.2615655
time: 4.53 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464454, upper bound: 1.2581830
time: 4.43 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2523196
time: 4.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464453, upper bound: 1.2634921
time: 6.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464454, upper bound: 1.2582422
time: 4.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838755, -9.2908850, -6.2820530, -2.6568856, 2.7001438
1: -6.8073831, -4.3517170, -6.8161888, -4.3487253, -2.3413434, 2.3515642
2: -8.7954197, -6.4763618, -8.8189726, -6.4719353, -2.2854633, 2.2904096
3: -10.1369629, -7.5283861, -10.1729803, -7.5255833, -2.2861013, 2.3208952
4: -5.0075345, -2.5014138, -5.0099111, -2.4916658, -2.5158687, 2.5084972
5: -5.4072838, -2.9503393, -5.4643207, -2.9493170, -2.3631287, 2.4242306
6: -13.6888151, -10.7062531, -13.6947041, -10.6661835, -3.0226316, 2.9884510
7: 3.2560353, 5.0214629, 3.2535262, 5.0367031, -1.7806678, 1.7679367
8: -4.4790020, -1.5509005, -4.5066218, -1.5477891, -2.5734305, 2.5701761
9: -2.3410516, 0.1115841, -2.3447247, 0.1304879, -2.4715395, 2.4563088

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2621060
time: 4.27 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562553, upper bound: 1.2568454
time: 4.29 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838755, -9.3378448, -6.2299337, -2.7090049, 2.7581334
1: -6.8073831, -4.3517170, -6.8579912, -4.3212600, -2.3704638, 2.3996291
2: -8.7954197, -6.4763618, -8.8412304, -6.4588165, -2.2982106, 2.3262544
3: -10.1369629, -7.5283861, -10.2079659, -7.4975381, -2.3149014, 2.3491902
4: -5.0075345, -2.5014138, -5.0471888, -2.4586282, -2.5489063, 2.5457749
5: -5.4072838, -2.9503393, -5.4968271, -2.9152298, -2.3995628, 2.4541757
6: -13.6888151, -10.7062531, -13.7874107, -10.6013489, -3.0874662, 3.0811577
7: 3.2560353, 5.0214629, 3.2235408, 5.0704460, -1.8144107, 1.7979221
8: -4.4790020, -1.5509005, -4.5584888, -1.5123181, -2.6095433, 2.6080844
9: -2.3410516, 0.1115841, -2.3868537, 0.1671182, -2.5081697, 2.4984379

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2674136
time: 4.41 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2615637
time: 4.38 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2847652, -6.2505021, -9.2543821, -6.3069811, -2.7026906, 2.7147949
1: -6.8429656, -4.3275976, -6.7960587, -4.3441381, -2.4118776, 2.4358416
2: -8.8085737, -6.4782786, -8.7728930, -6.5014582, -2.2736268, 2.2701187
3: -10.1603298, -7.5201378, -10.0982494, -7.5458784, -2.2920318, 2.2520666
4: -5.0334201, -2.4755089, -4.9843497, -2.5036590, -2.5297611, 2.5088408
5: -5.4245520, -2.9427199, -5.3808002, -2.9938142, -2.3972454, 2.3804874
6: -13.7616920, -10.6454554, -13.6688566, -10.6658096, -3.0958824, 3.0234013
7: 3.2425385, 5.0531502, 3.2870603, 5.0198431, -1.7773046, 1.7660899
8: -4.5231619, -1.5354137, -4.4600677, -1.5691185, -2.5864058, 2.5478508
9: -2.3691497, 0.1425672, -2.3351562, 0.0981163, -2.4672661, 2.4777234

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2517547
time: 4.56 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2517564
time: 4.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2929640, -6.2494798, -9.2866030, -6.2992134, -2.7226243, 2.7461970
1: -6.8476877, -4.3254099, -6.8183289, -4.3343110, -2.4288845, 2.4593005
2: -8.8160191, -6.4737797, -8.8025217, -6.4797010, -2.3002253, 2.2981327
3: -10.1710749, -7.5164533, -10.1429987, -7.5239100, -2.3255458, 2.2899942
4: -5.0392170, -2.4706659, -5.0094333, -2.4817386, -2.5574784, 2.5387673
5: -5.4359522, -2.9382517, -5.4285107, -2.9640732, -2.4388103, 2.4061961
6: -13.7662506, -10.6425905, -13.6904202, -10.6507959, -3.1154547, 3.0478296
7: 3.2330379, 5.0539656, 3.2483821, 5.0232911, -1.7902532, 1.8055835
8: -4.5272059, -1.5254412, -4.4855065, -1.5299273, -2.6123772, 2.5859199
9: -2.3735921, 0.1474540, -2.3557954, 0.1167918, -2.4903841, 2.5032494

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464423, upper bound: 1.2576316
time: 4.51 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2615638
time: 4.90 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.3296461, -6.2309604, -9.2543821, -6.3069811, -2.7364895, 2.7341511
1: -6.8532681, -4.3234448, -6.7960587, -4.3441381, -2.4225550, 2.4398336
2: -8.8337879, -6.4633064, -8.7728930, -6.5014582, -2.2984128, 2.2847979
3: -10.1972008, -7.5012407, -10.0982494, -7.5458784, -2.3193967, 2.2707567
4: -5.0413690, -2.4634609, -4.9843497, -2.5036590, -2.5377100, 2.5208888
5: -5.4854393, -2.9197016, -5.3808002, -2.9938142, -2.4281850, 2.4035544
6: -13.7828550, -10.6042137, -13.6688566, -10.6658096, -3.1170454, 3.0646429
7: 3.2330408, 5.0696354, 3.2870603, 5.0198431, -1.7868023, 1.7825751
8: -4.5544519, -1.5222950, -4.4600677, -1.5691185, -2.5996766, 2.5612226
9: -2.3823972, 0.1622107, -2.3351562, 0.0981163, -2.4805136, 2.4973669

Time for backsubstitution: 12.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2517546
time: 4.62 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2517552
time: 4.21 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.3378448, -6.2299337, -9.2866030, -6.2992134, -2.7580252, 2.7655618
1: -6.8579912, -4.3212600, -6.8183289, -4.3343110, -2.4397120, 2.4632938
2: -8.8412304, -6.4588165, -8.8025217, -6.4797010, -2.3250198, 2.3128402
3: -10.2079659, -7.4975381, -10.1429987, -7.5239100, -2.3545802, 2.3087049
4: -5.0471888, -2.4586282, -5.0094333, -2.4817386, -2.5654502, 2.5508051
5: -5.4968271, -2.9152298, -5.4285107, -2.9640732, -2.4683981, 2.4292636
6: -13.7874107, -10.6013489, -13.6904202, -10.6507959, -3.1366148, 3.0890713
7: 3.2235408, 5.0704460, 3.2483821, 5.0232911, -1.7997503, 1.8220639
8: -4.5584888, -1.5123181, -4.4855065, -1.5299273, -2.6262650, 2.5992990
9: -2.3868537, 0.1671182, -2.3557954, 0.1167918, -2.5036454, 2.5229135

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2576317
time: 4.29 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2615638
time: 4.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396107, -9.2826748, -6.2830811, -2.6888409, 2.6886868
1: -6.8267608, -4.3343019, -6.8113551, -4.3509269, -2.3657632, 2.3636060
2: -8.7878790, -6.4858303, -8.8114977, -6.4764667, -2.2931895, 2.2756321
3: -10.1272211, -7.5233607, -10.1622047, -7.5292969, -2.2849755, 2.3157821
4: -5.0195069, -2.4906087, -5.0040612, -2.4965551, -2.5229518, 2.5134525
5: -5.3917093, -2.9466329, -5.4529486, -2.9537749, -2.3697219, 2.4008870
6: -13.7600489, -10.6568575, -13.6901283, -10.6690788, -3.0909700, 3.0332708
7: 3.2653728, 5.0516901, 3.2631536, 5.0358996, -1.7705269, 1.7885365
8: -4.5052967, -1.5559220, -4.5025253, -1.5578218, -2.5816121, 2.5798395
9: -2.3622141, 0.1295742, -2.3402143, 0.1255753, -2.4877894, 2.4697886

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517571, upper bound: 1.2581806
time: 4.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2529326
time: 4.45 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396107, -9.3296461, -6.2309604, -2.7131047, 2.7573204
1: -6.8267608, -4.3343019, -6.8532681, -4.3234448, -2.4474316, 2.4648194
2: -8.7878790, -6.4858303, -8.8337879, -6.4633064, -2.3171334, 2.3231213
3: -10.1272211, -7.5233607, -10.1972008, -7.5012407, -2.3107929, 2.3419633
4: -5.0195069, -2.4906087, -5.0413690, -2.4634609, -2.5560460, 2.5507603
5: -5.3917093, -2.9466329, -5.4854393, -2.9197016, -2.4223161, 2.4539781
6: -13.7600489, -10.6568575, -13.7828550, -10.6042137, -3.1558352, 3.1259975
7: 3.2653728, 5.0516901, 3.2330408, 5.0696354, -1.8042626, 1.8186493
8: -4.5052967, -1.5559220, -4.5544519, -1.5222950, -2.6168461, 2.6157174
9: -2.3622141, 0.1295742, -2.3823972, 0.1622107, -2.5244248, 2.5119715

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517573, upper bound: 1.2581810
time: 4.56 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2517574, upper bound: 1.2529326
time: 4.66 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317109, -9.2992420, -6.2874622, -2.7353683, 2.7378597
1: -6.8491974, -4.3243456, -6.8063827, -4.3399591, -2.4203668, 2.4504128
2: -8.8176308, -6.4634013, -8.7980671, -6.4865298, -2.3195033, 2.3117385
3: -10.1720524, -7.5003276, -10.1351118, -7.5269899, -2.3353925, 2.3102269
4: -5.0447407, -2.4684758, -4.9922600, -2.4915531, -2.5531876, 2.5237842
5: -5.4397163, -2.9162726, -5.4417200, -2.9707978, -2.4388380, 2.4298565
6: -13.7818317, -10.6415138, -13.6900377, -10.6245613, -3.1572704, 3.0485239
7: 3.2260985, 5.0552688, 3.2775731, 5.0363922, -1.8102937, 1.7776957
8: -4.5310316, -1.5154614, -4.4915714, -1.5560551, -2.6120815, 2.5999448
9: -2.3830624, 0.1482716, -2.3482521, 0.1177129, -2.5007753, 2.4965236

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576307, upper bound: 1.2576159
time: 4.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2523648
time: 4.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2944183, -6.2317109, -9.3314724, -6.2796717, -2.7416725, 2.7691908
1: -6.8491974, -4.3243456, -6.8286371, -4.3301435, -2.4346538, 2.4755912
2: -8.8176308, -6.4634013, -8.8276768, -6.4648428, -2.3342142, 2.3371420
3: -10.1720524, -7.5003276, -10.1799822, -7.5049653, -2.3456395, 2.3450086
4: -5.0447407, -2.4684758, -5.0174160, -2.4696631, -2.5750775, 2.5489402
5: -5.4397163, -2.9162726, -5.4894171, -2.9410391, -2.4439054, 2.4659235
6: -13.7818317, -10.6415138, -13.7115927, -10.6095581, -3.1722736, 3.0700788
7: 3.2260985, 5.0552688, 3.2389002, 5.0398235, -1.8137250, 1.8163686
8: -4.5310316, -1.5154614, -4.5169911, -1.5168557, -2.6441011, 2.6174204
9: -2.3830624, 0.1482716, -2.3689346, 0.1365036, -2.5195661, 2.5172062

Time for backsubstitution: 12.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2650140
time: 4.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2597475
time: 4.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 22.35 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2523218
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2562553, upper bound: 1.2562579
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2576310
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2562553, upper bound: 1.2615656
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2523064, upper bound: 1.2523198
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2621030, upper bound: 1.2562580
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2523064, upper bound: 1.2576320
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2621028, upper bound: 1.2615655
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2464454, upper bound: 1.2581830
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2523196
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2464453, upper bound: 1.2634921
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2464454, upper bound: 1.2582422
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2621060
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2562553, upper bound: 1.2568454
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2674136
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2615637
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2517547
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2517564
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2464423, upper bound: 1.2576316
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2615638
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2517546
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2517552
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2576317
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2615638
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2517571, upper bound: 1.2581806
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2529326
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2517573, upper bound: 1.2581810
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2517574, upper bound: 1.2529326
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576307, upper bound: 1.2576159
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2523648
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2650140
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.35
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2597475

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2137794, -6.3093643, -9.2378464, -6.3026524, -2.6011691, 2.6175478
1: -6.7853842, -4.3627400, -6.8010497, -4.3550100, -2.3139143, 2.3229129
2: -8.7641144, -6.5075798, -8.7863140, -6.4913125, -2.2189231, 2.2254651
3: -10.0927734, -7.5665693, -10.1252527, -7.5481896, -2.2176695, 2.2324338
4: -4.9767933, -2.5248652, -4.9961500, -2.5085464, -2.4682469, 2.4712849
5: -5.3564520, -3.0019968, -5.3921118, -2.9767818, -2.3079834, 2.3163731
6: -13.6516829, -10.7222109, -13.6687059, -10.7102346, -2.9414482, 2.9464951
7: 3.3016930, 5.0167184, 3.2726197, 5.0193686, -1.7176757, 1.7440987
8: -4.4508424, -1.6002913, -4.4711232, -1.5709162, -2.5022411, 2.4996443
9: -2.3110857, 0.0929170, -2.3270938, 0.1059012, -2.4169869, 2.4200108

Time for backsubstitution: 12.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464425, upper bound: 1.2464424
time: 6.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464425, upper bound: 1.2523189
time: 6.24 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.2460384, -6.3016338, -9.2460556, -6.3016310, -2.6329250, 2.6377287
1: -6.8058748, -4.3528070, -6.8058786, -4.3528051, -2.3356142, 2.3396919
2: -8.7937784, -6.4867759, -8.7937965, -6.4867706, -2.2469735, 2.2513986
3: -10.1359940, -7.5445051, -10.1360102, -7.5444994, -2.2539234, 2.2659345
4: -5.0019689, -2.5036554, -5.0019784, -2.5036502, -2.4983187, 2.4983230
5: -5.4034872, -2.9723353, -5.4034948, -2.9723291, -2.3331933, 2.3577144
6: -13.6732759, -10.7073421, -13.6732826, -10.7073383, -2.9659376, 2.9659405
7: 3.2630086, 5.0201755, 3.2629938, 5.0201759, -1.7571673, 1.7571816
8: -4.4752202, -1.5609097, -4.4752259, -1.5608859, -2.5391173, 2.5253992
9: -2.3315768, 0.1107858, -2.3315849, 0.1107922, -2.4423690, 2.4423709

Time for backsubstitution: 12.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523197, upper bound: 1.2464419
time: 5.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523197, upper bound: 1.2562579
time: 4.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.2137794, -6.3093643, -9.2847652, -6.2505021, -2.6533699, 2.6821270
1: -6.7853842, -4.3627400, -6.8429656, -4.3275976, -2.3429747, 2.3708394
2: -8.7641144, -6.5075798, -8.8085737, -6.4782786, -2.2315230, 2.2612143
3: -10.0927734, -7.5665693, -10.1603298, -7.5201378, -2.2465158, 2.2692456
4: -4.9767933, -2.5248652, -5.0334201, -2.4755089, -2.5012844, 2.5085549
5: -5.3564520, -3.0019968, -5.4245520, -2.9427199, -2.3445983, 2.3493099
6: -13.6516829, -10.7222109, -13.7616920, -10.6454554, -3.0062275, 3.0394812
7: 3.3016930, 5.0167184, 3.2425385, 5.0531502, -1.7514572, 1.7741799
8: -4.4508424, -1.6002913, -4.5231619, -1.5354137, -2.5383348, 2.5548062
9: -2.3110857, 0.0929170, -2.3691497, 0.1425672, -2.4536529, 2.4620667

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464423, upper bound: 1.2517537
time: 5.77 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2464423, upper bound: 1.2576333
time: 4.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.2460384, -6.3016338, -9.2929640, -6.2494798, -2.6850014, 2.7019119
1: -6.8058748, -4.3528070, -6.8476877, -4.3254099, -2.3646278, 2.3877606
2: -8.7937784, -6.4867759, -8.8160191, -6.4737797, -2.2595749, 2.2871704
3: -10.1359940, -7.5445051, -10.1710749, -7.5164533, -2.2827554, 2.3027043
4: -5.0019689, -2.5036554, -5.0392170, -2.4706659, -2.5313029, 2.5355616
5: -5.4034872, -2.9723353, -5.4359522, -2.9382517, -2.3697062, 2.3907146
6: -13.6732759, -10.7073421, -13.7662506, -10.6425905, -3.0306854, 3.0589085
7: 3.2630086, 5.0201755, 3.2330379, 5.0539656, -1.7909570, 1.7871375
8: -4.4752202, -1.5609097, -4.5272059, -1.5254412, -2.5751987, 2.5807440
9: -2.3315768, 0.1107858, -2.3735921, 0.1474540, -2.4790308, 2.4843779

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523194, upper bound: 1.2517561
time: 4.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2523194, upper bound: 1.2517550
time: 4.43 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 21.66 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2464425, upper bound: 1.2464424
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2464425, upper bound: 1.2523189
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2523197, upper bound: 1.2464419
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2523197, upper bound: 1.2562579
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2464423, upper bound: 1.2517537
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2464423, upper bound: 1.2576333
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2523194, upper bound: 1.2517561
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 21.66
Output dim: 7, lower bound: -1.2523194, upper bound: 1.2517550
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2523064, upper bound: 1.2523198
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2621030, upper bound: 1.2562580
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2523064, upper bound: 1.2576320
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2621028, upper bound: 1.2615655
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2464454, upper bound: 1.2581830
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2464455, upper bound: 1.2523196
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2464453, upper bound: 1.2634921
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2464454, upper bound: 1.2582422
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2621060
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2562553, upper bound: 1.2568454
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2674136
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2562552, upper bound: 1.2615637
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2517547
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2517564
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2464423, upper bound: 1.2576316
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2517541, upper bound: 1.2615638
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2517546
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2517552
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2576317
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576150, upper bound: 1.2615638
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2517571, upper bound: 1.2581806
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2517572, upper bound: 1.2529326
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2517573, upper bound: 1.2581810
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2517574, upper bound: 1.2529326
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576307, upper bound: 1.2576159
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576309, upper bound: 1.2523648
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2650140
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.66
Output dim: 7, lower bound: -1.2576312, upper bound: 1.2597475
Binary search (step 0): status=Status.UNKNOWN, k_low=4, k_high=12, k_mid=8, eps_mid=0.0312500, abs_max=1.783203125
rel_dist={7: [-1.2674345485483558, 1.2674366290257746]}

## Binary search (step 1) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682702, upper bound: 1.0712157
time: 4.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712136, upper bound: 1.0712123
time: 4.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.10 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.10
Output dim: 7, lower bound: -1.0682702, upper bound: 1.0712157
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.10
Output dim: 7, lower bound: -1.0712136, upper bound: 1.0712123

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.2475243, -6.2838678, -9.2732029, -6.2823186, -2.3580770, 2.3966422
1: -6.8073869, -4.3517122, -6.8153067, -4.3400154, -2.1541529, 2.1525097
2: -8.7954388, -6.4763551, -8.8009901, -6.4719234, -2.0771532, 2.0873444
3: -10.1369781, -7.5283747, -10.1414671, -7.5154095, -2.0440259, 2.0342274
4: -5.0075459, -2.5014083, -5.0122819, -2.4875441, -2.4071321, 2.3958864
5: -5.4072948, -2.9503248, -5.4231334, -2.9450729, -2.1662798, 2.1756904
6: -13.6888247, -10.7062445, -13.6997213, -10.6703844, -3.0030622, 2.9934769
7: 3.2560186, 5.0214634, 3.2468052, 5.0234723, -1.7072635, 1.7088827
8: -4.4790077, -1.5508766, -4.4856310, -1.5312586, -2.3050761, 2.2931564
9: -2.3410625, 0.1115901, -2.3563323, 0.1154325, -2.3779292, 2.3905540

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682137, upper bound: 1.0673031
time: 4.25 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682678, upper bound: 1.0712100
time: 4.07 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.2944317, -6.2317004, -9.2880840, -6.2814517, -2.4220829, 2.4581695
1: -6.8492031, -4.3243413, -6.8198385, -4.3332157, -2.2141099, 2.2383270
2: -8.8176489, -6.4633946, -8.8041725, -6.4693041, -2.1152039, 2.1156573
3: -10.1720686, -7.5003147, -10.1440058, -7.5077810, -2.0889781, 2.0648527
4: -5.0447531, -2.4684689, -5.0150065, -2.4794931, -2.4581022, 2.4646630
5: -5.4397259, -2.9162605, -5.4323215, -2.9420645, -2.2224417, 2.2231569
6: -13.7818432, -10.6415081, -13.7059317, -10.6497011, -3.1321421, 3.0644236
7: 3.2260804, 5.0552716, 3.2414026, 5.0245962, -1.7313018, 1.7483954
8: -4.5310378, -1.5154338, -4.4893532, -1.5199156, -2.3714013, 2.3299689
9: -2.3830755, 0.1482754, -2.3652916, 0.1176102, -2.4216161, 2.4369264

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0711493, upper bound: 1.0673028
time: 4.12 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712111, upper bound: 1.0712103
time: 4.21 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.01 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.01
Output dim: 7, lower bound: -1.0682137, upper bound: 1.0673031
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.01
Output dim: 7, lower bound: -1.0682678, upper bound: 1.0712100
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.01
Output dim: 7, lower bound: -1.0711493, upper bound: 1.0673028
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.01
Output dim: 7, lower bound: -1.0712111, upper bound: 1.0712103

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.2472305, -6.2874746, -9.2717285, -6.3000827, -2.3399358, 2.3910804
1: -6.8070822, -4.3519311, -6.8137965, -4.3411098, -2.1527762, 2.1507559
2: -8.7951088, -6.4784713, -8.7993622, -6.4823246, -2.0613241, 2.0771904
3: -10.1367846, -7.5316496, -10.1404829, -7.5315390, -2.0255961, 2.0258403
4: -5.0064144, -2.5018604, -5.0067153, -2.4897864, -2.4011068, 2.3885441
5: -5.4065299, -2.9547923, -5.4193311, -2.9670753, -2.1434650, 2.1670280
6: -13.6856689, -10.7064657, -13.6842079, -10.6714811, -2.9986734, 2.9777422
7: 3.2574344, 5.0212040, 3.2537766, 5.0221734, -1.7042494, 1.7015047
8: -4.4782467, -1.5529099, -4.4818077, -1.5412617, -2.2903581, 2.2830174
9: -2.3391385, 0.1114287, -2.3468618, 0.1146251, -2.3721232, 2.3788428

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
time: 9.93 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0672970
time: 3.96 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.2475204, -6.2838802, -9.3165836, -6.2805290, -2.3557467, 2.4316814
1: -6.8073854, -4.3517132, -6.8241062, -4.3369713, -2.1573014, 2.1616893
2: -8.7954378, -6.4763589, -8.8245211, -6.4674730, -2.0882578, 2.1047645
3: -10.1369781, -7.5283875, -10.1774616, -7.5126009, -2.0471659, 2.0669518
4: -5.0075436, -2.5014102, -5.0146775, -2.4777417, -2.4138322, 2.4007187
5: -5.4072914, -2.9503343, -5.4802094, -2.9440508, -2.1633730, 2.2177353
6: -13.6888208, -10.7062464, -13.7054758, -10.6302719, -3.0429511, 2.9992294
7: 3.2560215, 5.0214624, 3.2442985, 5.0387058, -1.7222493, 1.7101383
8: -4.4790053, -1.5508809, -4.5132627, -1.5281806, -2.3133078, 2.3172696
9: -2.3410594, 0.1115897, -2.3600078, 0.1343310, -2.3941517, 2.3957171

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
time: 6.70 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0712034
time: 6.55 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.2941370, -6.2353139, -9.2866096, -6.2992125, -2.4039683, 2.4516263
1: -6.8488955, -4.3245544, -6.8183298, -4.3343120, -2.2127113, 2.2365670
2: -8.8173208, -6.4655037, -8.8025351, -6.4796982, -2.0994005, 2.1054931
3: -10.1718693, -7.5035925, -10.1430130, -7.5239096, -2.0705276, 2.0564723
4: -5.0436263, -2.4689126, -5.0094404, -2.4817369, -2.4520645, 2.4573112
5: -5.4389658, -2.9207246, -5.4285173, -2.9640684, -2.1996193, 2.2145879
6: -13.7786751, -10.6417246, -13.6904221, -10.6508045, -3.1278706, 3.0486975
7: 3.2274928, 5.0550098, 3.2483706, 5.0232916, -1.7282867, 1.7410187
8: -4.5302668, -1.5174665, -4.4855094, -1.5299158, -2.3566585, 2.3198276
9: -2.3811486, 0.1481103, -2.3557992, 0.1167973, -2.4158130, 2.4251943

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641757, upper bound: 1.0644659
time: 5.22 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0711423, upper bound: 1.0672961
time: 4.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.2944317, -6.2317157, -9.3314762, -6.2796683, -2.4197326, 2.4614282
1: -6.8492026, -4.3243408, -6.8286390, -4.3301449, -2.2172303, 2.2475286
2: -8.8176470, -6.4633989, -8.8276939, -6.4648399, -2.1263595, 2.1330533
3: -10.1720676, -7.5003271, -10.1799936, -7.5049677, -2.0920730, 2.0975835
4: -5.0447493, -2.4684708, -5.0174232, -2.4696620, -2.4623170, 2.4694800
5: -5.4397240, -2.9162717, -5.4894218, -2.9410336, -2.2195354, 2.2502449
6: -13.7818384, -10.6415100, -13.7115955, -10.6095638, -3.1412582, 3.0693445
7: 3.2260828, 5.0552711, 3.2388897, 5.0398245, -1.7462873, 1.7496588
8: -4.5310364, -1.5154424, -4.5169940, -1.5168438, -2.3727589, 2.3541226
9: -2.3830719, 0.1482751, -2.3689384, 0.1365067, -2.4378452, 2.4419322

Time for backsubstitution: 12.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642528, upper bound: 1.0683917
time: 4.47 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0712040, upper bound: 1.0712026
time: 4.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0682055, upper bound: 1.0672970
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0682595, upper bound: 1.0712034
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0641757, upper bound: 1.0644659
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0711423, upper bound: 1.0672961
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0642528, upper bound: 1.0683917
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.88
Output dim: 7, lower bound: -1.0712040, upper bound: 1.0712026

## BFS IS instance: IS_A1_B1_A1

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

Time for backsubstitution: 13.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0616529
time: 5.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
time: 11.64 seconds

## BFS IS instance: IS_A1_B1_A2

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

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0654866, upper bound: 1.0603221
time: 6.34 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0654866, upper bound: 1.0603224
time: 5.88 seconds

## BFS IS instance: IS_A1_B2_A1

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

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0655506
time: 6.58 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
time: 6.46 seconds

## BFS IS instance: IS_A1_B2_A2

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

Time for backsubstitution: 12.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0655513, upper bound: 1.0642501
time: 6.09 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0655513, upper bound: 1.0642502
time: 4.85 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.2612019, -6.2432113, -9.2730198, -6.3009334, -2.3654146, 2.4215550
1: -6.8264599, -4.3345137, -6.8103247, -4.3379498, -2.1866779, 2.2190661
2: -8.7875547, -6.4879451, -8.7901115, -6.4873118, -2.0630188, 2.0720413
3: -10.1270275, -7.5266290, -10.1250753, -7.5296445, -2.0184455, 2.0150645
4: -5.0183964, -2.4910529, -4.9997492, -2.4899588, -2.4159002, 2.4182973
5: -5.3909531, -2.9510908, -5.4094772, -2.9716132, -2.1432142, 2.1621218
6: -13.7568865, -10.6570749, -13.6827965, -10.6556654, -3.1012211, 3.0201001
7: 3.2667809, 5.0514269, 3.2642560, 5.0219421, -1.6863732, 1.7206050
8: -4.5045271, -1.5579486, -4.4786668, -1.5465503, -2.3036213, 2.2739334
9: -2.3602879, 0.1294141, -2.3482718, 0.1086478, -2.3847017, 2.3958268

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641756, upper bound: 1.0616526
time: 5.10 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0641758, upper bound: 1.0616527
time: 5.05 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.2941246, -6.2353144, -9.2866039, -6.2992144, -2.3982010, 2.4494126
1: -6.8488922, -4.3245587, -6.8183289, -4.3343129, -2.2110343, 2.2383482
2: -8.8173027, -6.4655080, -8.8025312, -6.4796991, -2.0919914, 2.1041825
3: -10.1718550, -7.5035982, -10.1430073, -7.5239115, -2.0562563, 2.0564637
4: -5.0436172, -2.4689188, -5.0094380, -2.4817376, -2.4463563, 2.4528704
5: -5.4389563, -2.9207311, -5.4285135, -2.9640703, -2.1661549, 2.2120450
6: -13.7786684, -10.6417303, -13.6904230, -10.6508055, -3.1278629, 3.0486927
7: 3.2275081, 5.0550079, 3.2483745, 5.0232911, -1.7282743, 1.7412809
8: -4.5302606, -1.5174894, -4.4855084, -1.5299206, -2.3470130, 2.2995689
9: -2.3811395, 0.1481059, -2.3557966, 0.1167954, -2.4151306, 2.4229889

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0683165, upper bound: 1.0603229
time: 5.05 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0683165, upper bound: 1.0672960
time: 5.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2614946, -6.2396154, -9.3178902, -6.2814026, -2.3811598, 2.4313593
1: -6.8267603, -4.3343039, -6.8206282, -4.3337770, -2.1911912, 2.2299900
2: -8.7878799, -6.4858322, -8.8152714, -6.4724345, -2.0899873, 2.0996139
3: -10.1272221, -7.5233636, -10.1620226, -7.5107231, -2.0400302, 2.0561583
4: -5.0195060, -2.4906094, -5.0076981, -2.4778693, -2.4264522, 2.4276924
5: -5.3917093, -2.9466360, -5.4704003, -2.9485853, -2.1631122, 2.1861296
6: -13.7600451, -10.6568575, -13.7039709, -10.6144218, -3.1085529, 3.0371971
7: 3.2653728, 5.0516911, 3.2547779, 5.0384846, -1.7043920, 1.7292354
8: -4.5052958, -1.5559258, -4.5101628, -1.5334864, -2.3188324, 2.3082325
9: -2.3622122, 0.1295754, -2.3613834, 0.1283234, -2.4067140, 2.4125953

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642527, upper bound: 1.0655497
time: 4.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0642529, upper bound: 1.0655497
time: 4.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.2944174, -6.2317166, -9.3314743, -6.2796707, -2.4139638, 2.4592159
1: -6.8491979, -4.3243456, -6.8286366, -4.3301458, -2.2155538, 2.2494388
2: -8.8176317, -6.4634037, -8.8276892, -6.4648418, -2.1189799, 2.1317394
3: -10.1720505, -7.5003338, -10.1799908, -7.5049691, -2.0778008, 2.0975747
4: -5.0447388, -2.4684775, -5.0174198, -2.4696631, -2.4566226, 2.4647241
5: -5.4397149, -2.9162762, -5.4894185, -2.9410353, -2.1860585, 2.2353370
6: -13.7818308, -10.6415148, -13.7115955, -10.6095657, -3.1385813, 3.0700808
7: 3.2260985, 5.0552702, 3.2388935, 5.0398245, -1.7462752, 1.7499213
8: -4.5310297, -1.5154634, -4.5169945, -1.5168476, -2.3622184, 2.3339376
9: -2.3830605, 0.1482708, -2.3689363, 0.1365052, -2.4371605, 2.4397368

Time for backsubstitution: 12.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0683934, upper bound: 1.0642489
time: 4.68 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0683934, upper bound: 1.0712034
time: 5.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.52 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0616529
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0613517, upper bound: 1.0644662
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0654866, upper bound: 1.0603221
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0654866, upper bound: 1.0603224
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0655506
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0614172, upper bound: 1.0683928
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0655513, upper bound: 1.0642501
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0655513, upper bound: 1.0642502
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0641756, upper bound: 1.0616526
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0641758, upper bound: 1.0616527
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0683165, upper bound: 1.0603229
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0683165, upper bound: 1.0672960
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0642527, upper bound: 1.0655497
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0642529, upper bound: 1.0655497
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0683934, upper bound: 1.0642489
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.52
Output dim: 7, lower bound: -1.0683934, upper bound: 1.0712034

## BFS IS instance: IS_A1_B1_A1_B1

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

Time for backsubstitution: 12.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575197, upper bound: 1.0616533
time: 5.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0616539
time: 5.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

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

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
time: 4.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
time: 5.30 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2472153, -6.2874780, -9.2394581, -6.3078384, -2.3285203, 2.3570085
1: -6.8070784, -4.3519344, -6.7913914, -4.3509779, -2.1414914, 2.1286092
2: -8.7950916, -6.4784746, -8.7697048, -6.5040741, -2.0408678, 2.0460970
3: -10.1367664, -7.5316544, -10.0957041, -7.5537267, -2.0032768, 1.9790567
4: -5.0064049, -2.5018668, -4.9815893, -2.5117247, -2.3729954, 2.3582101
5: -5.4065199, -2.9547989, -5.3716292, -2.9967947, -2.1111808, 2.1161327
6: -13.6856623, -10.7064714, -13.6626368, -10.6864977, -2.9757481, 2.9556966
7: 3.2574492, 5.0212035, 3.2925992, 5.0187225, -1.7007220, 1.6614501
8: -4.4782410, -1.5529308, -4.4562998, -1.5805445, -2.2512369, 2.2517231
9: -2.3391280, 0.1114235, -2.3263001, 0.0959667, -2.3496327, 2.3574214

Time for backsubstitution: 12.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0603241
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0603254
time: 5.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2472153, -6.2874780, -9.2717133, -6.3000836, -2.3341637, 2.3853641
1: -6.8070784, -4.3519344, -6.8137922, -4.3411126, -2.1544662, 2.1525347
2: -8.7950916, -6.4784746, -8.7993469, -6.4823298, -2.0539355, 2.0698602
3: -10.1367664, -7.5316544, -10.1404686, -7.5315423, -2.0113192, 2.0115910
4: -5.0064049, -2.5018668, -5.0067048, -2.4897921, -2.3953986, 2.3828521
5: -5.4065199, -2.9547989, -5.4193215, -2.9670818, -2.1100502, 2.1337295
6: -13.6856623, -10.7064714, -13.6842012, -10.6714878, -3.0141745, 2.9777298
7: 3.2574492, 5.0212035, 3.2537913, 5.0221725, -1.7045312, 1.7017819
8: -4.4782410, -1.5529308, -4.4818039, -1.5412836, -2.2700138, 2.2625601
9: -2.3391280, 0.1114235, -2.3468516, 0.1146190, -2.3701253, 2.3766921

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0641397
time: 5.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0603254
time: 5.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0654859
time: 8.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0618686
time: 4.99 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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

Time for backsubstitution: 12.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0683163
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0647016
time: 4.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838802, -9.2843037, -6.2883091, -2.3443017, 2.3954470
1: -6.8073807, -4.3517170, -6.8017163, -4.3468294, -2.1460123, 2.1393611
2: -8.7954197, -6.4763637, -8.7948885, -6.4891748, -2.0678220, 2.0737283
3: -10.1369638, -7.5283914, -10.1325645, -7.5348530, -2.0249023, 2.0200906
4: -5.0075321, -2.5014157, -4.9894800, -2.4996521, -2.3857393, 2.3704391
5: -5.4072838, -2.9503403, -5.4325218, -2.9737873, -2.1310744, 2.1543956
6: -13.6888123, -10.7062540, -13.6838942, -10.6452789, -3.0200262, 2.9728432
7: 3.2560372, 5.0214629, 3.2831249, 5.0352688, -1.7187467, 1.6700435
8: -4.4790025, -1.5509024, -4.4877739, -1.5674701, -2.2742105, 2.2747386
9: -2.3410499, 0.1115841, -2.3393960, 0.1155555, -2.3715782, 2.3743477

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0641724
time: 6.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0605586
time: 5.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2475071, -6.2838802, -9.3165703, -6.2805305, -2.3499742, 2.4259028
1: -6.8073807, -4.3517170, -6.8241034, -4.3369765, -2.1589990, 2.1635969
2: -8.7954197, -6.4763637, -8.8245039, -6.4674797, -2.0809007, 2.0974419
3: -10.1369638, -7.5283914, -10.1774464, -7.5126066, -2.0328898, 2.0527086
4: -5.0075321, -2.5014157, -5.0146685, -2.4777493, -2.4081626, 2.3950276
5: -5.4072838, -2.9503403, -5.4801998, -2.9440560, -2.1299472, 2.1843667
6: -13.6888123, -10.7062540, -13.7054682, -10.6302776, -3.0585346, 2.9992142
7: 3.2560372, 5.0214629, 3.2443151, 5.0387049, -1.7225409, 1.7104161
8: -4.4790025, -1.5509024, -4.5132551, -1.5282030, -2.2929811, 2.2968879
9: -2.3410499, 0.1115841, -2.3599992, 0.1343240, -2.3921614, 2.3935776

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0679429
time: 4.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0641411
time: 5.09 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.2611666, -6.2432690, -9.2324219, -6.3033462, -2.3571954, 2.3617203
1: -6.8264036, -4.3345780, -6.7977562, -4.3564796, -2.1455035, 2.1358149
2: -8.7873144, -6.4879761, -8.7813454, -6.4944072, -2.0496798, 2.0363402
3: -10.1268482, -7.5266395, -10.1180573, -7.5506964, -1.9953160, 2.0085187
4: -5.0183353, -2.4910719, -4.9922562, -2.5118928, -2.3785181, 2.3713298
5: -5.3909001, -2.9511545, -5.3844738, -2.9798348, -2.1050420, 2.1255927
6: -13.7568407, -10.6571560, -13.6656504, -10.7122030, -3.0436153, 2.9837651
7: 3.2667894, 5.0514102, 3.2790308, 5.0188293, -1.6859848, 1.7119504
8: -4.5045238, -1.5579839, -4.4683371, -1.5775962, -2.2722845, 2.2650795
9: -2.3602407, 0.1294034, -2.3240569, 0.1026553, -2.3797312, 2.3700883

Time for backsubstitution: 12.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0616517
time: 4.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0616530
time: 5.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2612019, -6.2432113, -9.2793503, -6.2511954, -2.3690071, 2.3937702
1: -6.8264599, -4.3345137, -6.8397512, -4.3290553, -2.2165265, 2.2254305
2: -8.7875547, -6.4879451, -8.8036289, -6.4813433, -2.0720868, 2.0808210
3: -10.1270275, -7.5266290, -10.1531420, -7.5225763, -2.0208664, 2.0416181
4: -5.0183964, -2.4910529, -5.0295467, -2.4788165, -2.4262638, 2.4222383
5: -5.3909531, -2.9510908, -5.4169016, -2.9457836, -2.1476870, 2.1656830
6: -13.7568865, -10.6570749, -13.7586517, -10.6474028, -3.1012144, 3.0900669
7: 3.2667809, 5.0514269, 3.2488656, 5.0526061, -1.7115965, 1.7288145
8: -4.5045271, -1.5579486, -4.5204182, -1.5420556, -2.3030877, 2.3091197
9: -2.3602879, 0.1294141, -2.3661506, 0.1393294, -2.4112411, 2.4071109

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0616516
time: 4.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0616530
time: 6.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2941246, -6.2353144, -9.2543697, -6.3069820, -2.3928323, 2.4154036
1: -6.8488922, -4.3245587, -6.7960558, -4.3441439, -2.2012715, 2.2143235
2: -8.8173027, -6.4655080, -8.7728901, -6.5014601, -2.0789189, 2.0743794
3: -10.1718550, -7.5035982, -10.0982475, -7.5458846, -2.0481896, 2.0097330
4: -5.0436172, -2.4689188, -4.9843473, -2.5036645, -2.4161348, 2.4268780
5: -5.4389563, -2.9207311, -5.3807983, -2.9938171, -2.1639853, 2.1637874
6: -13.7786684, -10.6417303, -13.6688547, -10.6658192, -3.1064816, 3.0214076
7: 3.2275081, 5.0550079, 3.2870646, 5.0198417, -1.7247608, 1.7011601
8: -4.5302606, -1.5174894, -4.4600649, -1.5691271, -2.3092628, 2.2885814
9: -2.3811395, 0.1481059, -2.3351502, 0.0981146, -2.3932862, 2.4038763

Time for backsubstitution: 12.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0603229
time: 4.55 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0603254
time: 6.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2941246, -6.2353144, -9.2865915, -6.2992144, -2.3981977, 2.4458230
1: -6.8488922, -4.3245587, -6.8183270, -4.3343153, -2.2146740, 2.2383425
2: -8.8173027, -6.4655080, -8.8025188, -6.4797039, -2.0919867, 2.0981839
3: -10.1718550, -7.5035982, -10.1429977, -7.5239162, -2.0562525, 2.0421960
4: -5.0436172, -2.4689188, -5.0094309, -2.4817431, -2.4463482, 2.4518614
5: -5.4389563, -2.9207311, -5.4285078, -2.9640746, -2.1661463, 2.1812356
6: -13.7786684, -10.6417303, -13.6904173, -10.6508045, -3.1278639, 3.0486870
7: 3.2275081, 5.0550079, 3.2483869, 5.0232902, -1.7285728, 1.7412710
8: -4.5302606, -1.5174894, -4.4855032, -1.5299358, -2.3366137, 2.2995622
9: -2.3811395, 0.1481059, -2.3557885, 0.1167917, -2.4138918, 2.4229794

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641396
time: 5.32 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641411
time: 4.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2614594, -6.2396727, -9.2772512, -6.2837772, -2.3729696, 2.3716061
1: -6.8267035, -4.3343658, -6.8080626, -4.3523951, -2.1499939, 2.1466496
2: -8.7876396, -6.4858646, -8.8065348, -6.4795561, -2.0766087, 2.0639651
3: -10.1270418, -7.5233760, -10.1549950, -7.5318170, -2.0169594, 2.0454683
4: -5.0194478, -2.4906321, -5.0001531, -2.4998963, -2.3893995, 2.3835449
5: -5.3916559, -2.9466987, -5.4453173, -2.9568315, -2.1249423, 2.1497061
6: -13.7600021, -10.6569357, -13.6870728, -10.6710453, -3.0507507, 3.0008974
7: 3.2653818, 5.0516734, 3.2695661, 5.0353632, -1.7039905, 1.7205826
8: -4.5052910, -1.5559616, -4.4997439, -1.5645065, -2.2874975, 2.2911434
9: -2.3621655, 0.1295614, -2.3371677, 0.1223143, -2.4017301, 2.3868420

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0654859
time: 4.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0616551
time: 5.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2614946, -6.2396154, -9.3242321, -6.2316594, -2.3848071, 2.4263282
1: -6.8267603, -4.3343039, -6.8500504, -4.3249011, -2.2209206, 2.2364321
2: -8.7878799, -6.4858322, -8.8288469, -6.4663630, -2.0989552, 2.1084626
3: -10.1272221, -7.5233636, -10.1900005, -7.5036821, -2.0424614, 2.0669751
4: -5.0195060, -2.4906094, -5.0374823, -2.4667635, -2.4386096, 2.4318242
5: -5.3917093, -2.9466360, -5.4777975, -2.9227684, -2.1675549, 2.1941028
6: -13.7600451, -10.6568575, -13.7798119, -10.6061602, -3.1128197, 3.1071625
7: 3.2653728, 5.0516911, 3.2393689, 5.0690937, -1.7295473, 1.7374728
8: -4.5052958, -1.5559258, -4.5517092, -1.5289421, -2.3210516, 2.3284409
9: -2.3622122, 0.1295754, -2.3793862, 0.1589589, -2.4332075, 2.4238815

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0654863
time: 4.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0616551
time: 5.56 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2944174, -6.2317166, -9.2992325, -6.2874637, -2.4085636, 2.4252067
1: -6.8491979, -4.3243456, -6.8063807, -4.3399653, -2.2057900, 2.2251322
2: -8.8176317, -6.4634037, -8.7980633, -6.4865317, -2.1058960, 2.1019754
3: -10.1720505, -7.5003338, -10.1351118, -7.5269966, -2.0686736, 2.0507708
4: -5.0447388, -2.4684775, -4.9922571, -2.4915574, -2.4237757, 2.4389017
5: -5.4397149, -2.9162762, -5.4417200, -2.9708002, -2.1839085, 2.1869717
6: -13.7818308, -10.6415148, -13.6900368, -10.6245737, -3.1136818, 3.0385017
7: 3.2260985, 5.0552702, 3.2775788, 5.0363922, -1.7427855, 1.7097702
8: -4.5310297, -1.5154634, -4.4915714, -1.5560641, -2.3244786, 2.3118892
9: -2.3830605, 0.1482708, -2.3482480, 0.1177118, -2.4152350, 2.4206638

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0641727
time: 4.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0605566
time: 5.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2944174, -6.2317166, -9.3314629, -6.2796726, -2.4139609, 2.4556258
1: -6.8491979, -4.3243456, -6.8286333, -4.3301487, -2.2192006, 2.2494335
2: -8.8176317, -6.4634037, -8.8276758, -6.4648442, -2.1189752, 2.1257498
3: -10.1720505, -7.5003338, -10.1799784, -7.5049744, -2.0777967, 2.0833066
4: -5.0447388, -2.4684775, -5.0174127, -2.4696670, -2.4566174, 2.4640312
5: -5.4397149, -2.9162762, -5.4894123, -2.9410398, -2.1860499, 2.2168207
6: -13.7818308, -10.6415148, -13.7115889, -10.6095686, -3.1489849, 3.0700741
7: 3.2260985, 5.0552702, 3.2389059, 5.0398235, -1.7465839, 1.7499115
8: -4.5310297, -1.5154634, -4.5169902, -1.5168648, -2.3521037, 2.3339319
9: -2.3830605, 0.1482708, -2.3689280, 0.1365007, -2.4359303, 2.4397278

Time for backsubstitution: 12.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641736
time: 6.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0605586
time: 5.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.88 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575197, upper bound: 1.0616533
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0616539
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0644684
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0603241
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0603254
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0641397
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0603254
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0654859
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0618686
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0683163
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0575175, upper bound: 1.0647016
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0641724
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616525, upper bound: 1.0605586
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0679429
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0616529, upper bound: 1.0641411
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0616517
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0616530
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0616516
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0616530
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0603229
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0603254
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641396
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641411
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0654859
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603252, upper bound: 1.0616551
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0654863
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0603254, upper bound: 1.0616551
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0641727
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644658, upper bound: 1.0605566
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0641736
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.88
Output dim: 7, lower bound: -1.0644661, upper bound: 1.0605586

## BFS IS instance: IS_A1_B1_A1_B1_A1

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

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575177, upper bound: 1.0575177
time: 4.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0575199, upper bound: 1.0616531
time: 5.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

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

Time for backsubstitution: 12.46 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=4, k_high=7, k_mid=5, eps_mid=0.0195312, abs_max=1.7166380882263184
rel_dist={7: [-1.071222731717584, 1.071219330481989]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677270, upper bound: 0.9703685
time: 4.96 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703687, upper bound: 0.9703706
time: 4.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.90 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.90
Output dim: 7, lower bound: -0.9677270, upper bound: 0.9703685
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.90
Output dim: 7, lower bound: -0.9703687, upper bound: 0.9703706

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.2475243, -6.2838678, -9.2691727, -6.2825584, -2.2550278, 2.2894192
1: -6.8073869, -4.3517122, -6.8140688, -4.3418493, -2.0785203, 2.0775902
2: -8.7954388, -6.4763551, -8.8001232, -6.4726253, -2.0067639, 2.0168219
3: -10.1369781, -7.5283747, -10.1407738, -7.5174689, -1.9555321, 1.9471872
4: -5.0075459, -2.5014083, -5.0115399, -2.4897180, -2.3383861, 2.3288889
5: -5.4072948, -2.9503248, -5.4206524, -2.9459004, -2.0853267, 2.0936601
6: -13.6888247, -10.7062445, -13.6980362, -10.6759834, -2.9075003, 2.9112501
7: 3.2560186, 5.0214634, 3.2482605, 5.0231638, -1.6590190, 1.6596178
8: -4.4790077, -1.5508766, -4.4846106, -1.5343280, -2.2096457, 2.1997130
9: -2.3410625, 0.1115901, -2.3539388, 0.1148372, -2.3244762, 2.3351994

Time for backsubstitution: 12.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677228, upper bound: 0.9674741
time: 4.47 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677252, upper bound: 0.9703660
time: 6.42 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.2944317, -6.2317004, -9.2880783, -6.2814493, -2.3153400, 2.3539512
1: -6.8492031, -4.3243413, -6.8198380, -4.3332176, -2.1425877, 2.1632328
2: -8.8176489, -6.4633946, -8.8041706, -6.4693060, -2.0459247, 2.0457377
3: -10.1720686, -7.5003147, -10.1440029, -7.5077844, -2.0026784, 1.9783673
4: -5.0447531, -2.4684689, -5.0150032, -2.4794962, -2.3931103, 2.3966680
5: -5.4397259, -2.9162605, -5.4323196, -2.9420655, -2.1407871, 2.1444025
6: -13.7818432, -10.6415081, -13.7059288, -10.6497087, -3.0461907, 2.9772706
7: 3.2260804, 5.0552716, 3.2414055, 5.0245957, -1.6829448, 1.7004983
8: -4.5310378, -1.5154338, -4.4893527, -1.5199194, -2.2783976, 2.2369311
9: -2.3830755, 0.1482754, -2.3652892, 0.1176105, -2.3685846, 2.3841877

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703482, upper bound: 0.9674751
time: 4.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703668, upper bound: 0.9703660
time: 4.47 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.82 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.82
Output dim: 7, lower bound: -0.9677228, upper bound: 0.9674741
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.82
Output dim: 7, lower bound: -0.9677252, upper bound: 0.9703660
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.82
Output dim: 7, lower bound: -0.9703482, upper bound: 0.9674751
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.82
Output dim: 7, lower bound: -0.9703668, upper bound: 0.9703660

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -9.2470398, -6.2897730, -9.2677021, -6.3003221, -2.2366376, 2.2815623
1: -6.8068876, -4.3520708, -6.8125629, -4.3429432, -2.0769472, 2.0757046
2: -8.7948952, -6.4798179, -8.7984915, -6.4830284, -1.9899988, 2.0048189
3: -10.1366606, -7.5337343, -10.1397934, -7.5335970, -1.9364767, 1.9365427
4: -5.0056949, -2.5021486, -5.0059738, -2.4919600, -2.3315392, 2.3209248
5: -5.4060407, -2.9576375, -5.4168501, -2.9679060, -2.0619879, 2.0821574
6: -13.6836576, -10.7066059, -13.6824980, -10.6770782, -2.9010592, 2.8949580
7: 3.2583365, 5.0210390, 3.2552314, 5.0218663, -1.6550932, 1.6520377
8: -4.4777560, -1.5542049, -4.4807935, -1.5443349, -2.1939616, 2.1878681
9: -2.3379121, 0.1113255, -2.3444655, 0.1140299, -2.3172469, 2.3230238

Time for backsubstitution: 12.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9624946, upper bound: 0.9655598
time: 4.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677165, upper bound: 0.9674689
time: 4.69 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -9.2475214, -6.2838807, -9.3125515, -6.2807603, -2.2505078, 2.3227265
1: -6.8073854, -4.3517137, -6.8228703, -4.3388157, -2.0816641, 2.0867584
2: -8.7954359, -6.4763594, -8.8236532, -6.4681802, -2.0159450, 2.0342488
3: -10.1369791, -7.5283880, -10.1767702, -7.5146613, -1.9564469, 1.9799094
4: -5.0075426, -2.5014100, -5.0139313, -2.4799249, -2.3450789, 2.3330808
5: -5.4072924, -2.9503365, -5.4777212, -2.9448819, -2.0801916, 2.1334362
6: -13.6888199, -10.7062454, -13.7037868, -10.6358767, -2.9473825, 2.9102173
7: 3.2560205, 5.0214634, 3.2457547, 5.0383978, -1.6740055, 1.6602900
8: -4.4790072, -1.5508819, -4.5122375, -1.5312505, -2.2160659, 2.2238154
9: -2.3410583, 0.1115898, -2.3576126, 0.1337343, -2.3406944, 2.3390827

Time for backsubstitution: 12.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
time: 4.40 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9677188, upper bound: 0.9703585
time: 4.40 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.2939491, -6.2376118, -9.2866020, -6.2992115, -2.2969742, 2.3448036
1: -6.8487015, -4.3246932, -6.8183289, -4.3343134, -2.1409874, 2.1613350
2: -8.8171101, -6.4668469, -8.8025331, -6.4796972, -2.0291862, 2.0337214
3: -10.1717405, -7.5056810, -10.1430130, -7.5239129, -1.9835992, 1.9677310
4: -5.0429096, -2.4691937, -5.0094390, -2.4817395, -2.3862538, 2.3886890
5: -5.4384804, -2.9235687, -5.4285164, -2.9640684, -2.1174355, 2.1329918
6: -13.7766581, -10.6418667, -13.6904211, -10.6508074, -3.0387955, 2.9609389
7: 3.2283926, 5.0548410, 3.2483730, 5.0232911, -1.6790180, 1.6929184
8: -4.5297723, -1.5187621, -4.4855094, -1.5299182, -2.2624898, 2.2250822
9: -2.3799224, 0.1480047, -2.3557961, 0.1167982, -2.3613605, 2.3719931

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9651095, upper bound: 0.9655560
time: 4.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703415, upper bound: 0.9674708
time: 5.50 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.2944298, -6.2317138, -9.3314753, -6.2796669, -2.3107972, 2.3572114
1: -6.8492012, -4.3243427, -6.8286376, -4.3301468, -2.1457033, 2.1724343
2: -8.8176470, -6.4634004, -8.8276930, -6.4648414, -2.0551553, 2.0631330
3: -10.1720676, -7.5003309, -10.1799936, -7.5049706, -2.0035398, 2.0110972
4: -5.0447493, -2.4684694, -5.0174212, -2.4696627, -2.3962264, 2.4008408
5: -5.4397225, -2.9162710, -5.4894204, -2.9410341, -2.1356535, 2.1692839
6: -13.7818365, -10.6415100, -13.7115965, -10.6095676, -3.0482836, 2.9761305
7: 3.2260857, 5.0552702, 3.2388916, 5.0398254, -1.6979296, 1.7011803
8: -4.5310335, -1.5154419, -4.5169940, -1.5168457, -2.2768903, 2.2610841
9: -2.3830712, 0.1482768, -2.3689353, 0.1365060, -2.3848133, 2.3879137

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6181
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9651409, upper bound: 0.9684626
time: 5.26 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9703601, upper bound: 0.9703589
time: 5.01 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 22.88 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9624946, upper bound: 0.9655598
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9677165, upper bound: 0.9674689
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9677188, upper bound: 0.9703585
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9651095, upper bound: 0.9655560
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9703415, upper bound: 0.9674708
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9651409, upper bound: 0.9684626
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.88
Output dim: 7, lower bound: -0.9703601, upper bound: 0.9703589

## BFS IS instance: IS_A1_B1_A2

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
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9658193, upper bound: 0.9622334
time: 5.05 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9658193, upper bound: 0.9622335
time: 4.72 seconds

## BFS IS instance: IS_A1_B2_A1

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

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9658417
time: 4.18 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
time: 4.38 seconds

## BFS IS instance: IS_A1_B2_A2

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

Time for backsubstitution: 14.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9658433, upper bound: 0.9651374
time: 4.22 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9658433, upper bound: 0.9703593
time: 4.73 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.2939339, -6.2376146, -9.2865963, -6.2992158, -2.2909031, 2.3425889
1: -6.8486962, -4.3246965, -6.8183270, -4.3343148, -2.1393123, 2.1628232
2: -8.8170938, -6.4668517, -8.8025284, -6.4797006, -2.0212331, 2.0324097
3: -10.1717253, -7.5056868, -10.1430054, -7.5239162, -1.9685779, 1.9677210
4: -5.0429010, -2.4692013, -5.0094357, -2.4817402, -2.3797112, 2.3842454
5: -5.4384708, -2.9235749, -5.4285116, -2.9640720, -2.0819006, 2.1304483
6: -13.7766514, -10.6418695, -13.6904182, -10.6508083, -3.0348959, 2.9766340
7: 3.2284083, 5.0548387, 3.2483788, 5.0232897, -1.6790051, 1.6931558
8: -4.5297680, -1.5187836, -4.4855080, -1.5299249, -2.2519474, 2.2033677
9: -2.3799126, 0.1479986, -2.3557930, 0.1167958, -2.3606806, 2.3696065

Time for backsubstitution: 12.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9684342, upper bound: 0.9622296
time: 4.47 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9684342, upper bound: 0.9622305
time: 8.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396183, -9.3154650, -6.2817225, -2.2715387, 2.3247886
1: -6.8267608, -4.3343024, -6.8191471, -4.3344345, -2.1191773, 2.1534967
2: -8.7878780, -6.4858317, -8.8130436, -6.4738412, -2.0175648, 2.0271726
3: -10.1272211, -7.5233669, -10.1587706, -7.5117803, -1.9505215, 1.9658380
4: -5.0195045, -2.4906106, -5.0059385, -2.4793954, -2.3585801, 2.3563209
5: -5.3917084, -2.9466376, -5.4669456, -2.9499936, -2.0777946, 2.1021955
6: -13.7600479, -10.6568584, -13.7025928, -10.6153221, -3.0143957, 2.9426818
7: 3.2653751, 5.0516906, 3.2576375, 5.0382423, -1.6558132, 1.6777464
8: -4.5052943, -1.5559268, -4.5089030, -1.5364871, -2.2205839, 2.2136345
9: -2.3622127, 0.1295733, -2.3599982, 0.1268563, -2.3519588, 2.3571911

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9651409, upper bound: 0.9658416
time: 4.23 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9651410, upper bound: 0.9658411
time: 4.88 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.2944155, -6.2317185, -9.3314667, -6.2796702, -2.3047261, 2.3549964
1: -6.8491979, -4.3243465, -6.8286362, -4.3301487, -2.1440268, 2.1740520
2: -8.8176308, -6.4634047, -8.8276863, -6.4648428, -2.0472326, 2.0618167
3: -10.1720514, -7.5003362, -10.1799879, -7.5049729, -1.9885187, 2.0110865
4: -5.0447383, -2.4684768, -5.0174179, -2.4696643, -2.3896966, 2.3949752
5: -5.4397144, -2.9162786, -5.4894176, -2.9410372, -2.1001058, 2.1543753
6: -13.7818298, -10.6415148, -13.7115936, -10.6095715, -3.0443869, 2.9918265
7: 3.2261000, 5.0552692, 3.2388978, 5.0398245, -1.6979160, 1.7014190
8: -4.5310321, -1.5154643, -4.5169911, -1.5168543, -2.2663484, 2.2394407
9: -2.3830609, 0.1482711, -2.3689320, 0.1365037, -2.3841281, 2.3855357

Time for backsubstitution: 12.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9684646, upper bound: 0.9651363
time: 5.23 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9684646, upper bound: 0.9703592
time: 5.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.81 seconds
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9658193, upper bound: 0.9622334
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9658193, upper bound: 0.9622335
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9658417
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9625187, upper bound: 0.9684632
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9658433, upper bound: 0.9651374
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9658433, upper bound: 0.9703593
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9684342, upper bound: 0.9622296
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9684342, upper bound: 0.9622305
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9651409, upper bound: 0.9658416
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9651410, upper bound: 0.9658411
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9684646, upper bound: 0.9651363
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 22.81
Output dim: 7, lower bound: -0.9684646, upper bound: 0.9703592

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2470255, -6.2897749, -9.2354193, -6.3080740, -2.2252321, 2.2474813
1: -6.8068833, -4.3520746, -6.7901187, -4.3528223, -2.0656619, 2.0535781
2: -8.7948799, -6.4798236, -8.7688322, -6.5047708, -1.9695401, 1.9737341
3: -10.1366463, -7.5337410, -10.0950098, -7.5558052, -1.9141631, 1.8897529
4: -5.0056829, -2.5021553, -4.9808388, -2.5139024, -2.3034277, 2.2905898
5: -5.4060316, -2.9576426, -5.3691535, -2.9976177, -2.0297160, 2.0312619
6: -13.6836510, -10.7066107, -13.6609421, -10.6920929, -2.8781295, 2.8640618
7: 3.2583513, 5.0210376, 3.2940898, 5.0184145, -1.6515653, 1.6119313
8: -4.4777541, -1.5542264, -4.4552689, -1.5836334, -2.1548195, 2.1565604
9: -2.3379030, 0.1113198, -2.3239124, 0.0953778, -2.2947598, 2.3016024

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9622298
time: 4.74 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9622334
time: 4.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2470255, -6.2897749, -9.2676868, -6.3003225, -2.2305665, 2.2755408
1: -6.8068833, -4.3520746, -6.8125582, -4.3429461, -2.0783334, 2.0771632
2: -8.7948799, -6.4798236, -8.7984753, -6.4830337, -1.9820728, 1.9969435
3: -10.1366463, -7.5337410, -10.1397762, -7.5336008, -1.9214530, 1.9215412
4: -5.0056829, -2.5021553, -5.0059624, -2.4919660, -2.3249998, 2.3144002
5: -5.4060316, -2.9576426, -5.4168415, -2.9679108, -2.0265203, 2.0467882
6: -13.6836510, -10.7066107, -13.6824932, -10.6770840, -2.9167471, 2.9106450
7: 3.2583513, 5.0210376, 3.2552476, 5.0218654, -1.6553507, 1.6522911
8: -4.4777541, -1.5542264, -4.4807868, -1.5443540, -2.1721306, 2.1659405
9: -2.3379030, 0.1113198, -2.3444557, 0.1140251, -2.3150616, 2.3207111

Time for backsubstitution: 12.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9622308
time: 4.49 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9646686
time: 4.50 seconds

## BFS IS instance: IS_A1_B2_A1_B1

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

Time for backsubstitution: 12.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9658177
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9629434
time: 5.18 seconds

## BFS IS instance: IS_A1_B2_A1_B2

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
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9684325
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9596188, upper bound: 0.9655597
time: 5.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2475061, -6.2838831, -9.2802639, -6.2885389, -2.2390714, 2.2864826
1: -6.8073816, -4.3517175, -6.8004460, -4.3486824, -2.0703740, 2.0644505
2: -8.7954206, -6.4763641, -8.7940178, -6.4898834, -1.9955068, 2.0032206
3: -10.1369629, -7.5283947, -10.1318684, -7.5369358, -1.9341884, 1.9330404
4: -5.0075331, -2.5014160, -4.9887247, -2.5018382, -2.3169899, 2.3028007
5: -5.4072819, -2.9503427, -5.4300380, -2.9746103, -2.0479074, 2.0700974
6: -13.6888123, -10.7062511, -13.6822271, -10.6508865, -2.9244514, 2.8793392
7: 3.2560368, 5.0214620, 3.2846184, 5.0349612, -1.6705022, 1.6201440
8: -4.4790010, -1.5509028, -4.4867339, -1.5705557, -2.1769485, 2.1782842
9: -2.3410478, 0.1115847, -2.3370078, 0.1149653, -2.3181233, 2.3177109

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9651059
time: 4.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9622335
time: 4.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2475061, -6.2838831, -9.3125372, -6.2807655, -2.2444382, 2.3166447
1: -6.8073816, -4.3517175, -6.8228664, -4.3388181, -2.0830588, 2.0883491
2: -8.7954206, -6.4763641, -8.8236370, -6.4681864, -2.0080490, 2.0263824
3: -10.1369629, -7.5283947, -10.1767521, -7.5146666, -1.9414234, 1.9649179
4: -5.0075331, -2.5014160, -5.0139217, -2.4799314, -2.3385811, 2.3265586
5: -5.4072819, -2.9503427, -5.4777117, -2.9448876, -2.0447121, 2.0979950
6: -13.6888123, -10.7062511, -13.7037802, -10.6358824, -2.9630699, 2.9259052
7: 3.2560368, 5.0214620, 3.2457714, 5.0383968, -1.6742733, 1.6605446
8: -4.4790010, -1.5509028, -4.5122318, -1.5312719, -2.1942520, 2.2019608
9: -2.3410478, 0.1115847, -2.3576019, 0.1337290, -2.3385234, 2.3367791

Time for backsubstitution: 12.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9675413
time: 4.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9622334
time: 4.52 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.2939339, -6.2376146, -9.2543659, -6.3069839, -2.2858381, 2.3085811
1: -6.8486962, -4.3246965, -6.7960558, -4.3441463, -2.1295505, 2.1390910
2: -8.8170938, -6.4668517, -8.7728901, -6.5014601, -2.0087056, 2.0026090
3: -10.1717253, -7.5056868, -10.0982466, -7.5458889, -1.9598875, 1.9209931
4: -5.0429010, -2.4692013, -4.9843464, -2.5036652, -2.3482838, 2.3582554
5: -5.4384708, -2.9235749, -5.3807983, -2.9938183, -2.0774412, 2.0821924
6: -13.7766514, -10.6418695, -13.6688528, -10.6658249, -3.0108147, 2.9300761
7: 3.2284083, 5.0548387, 3.2870665, 5.0198421, -1.6754913, 1.6530595
8: -4.5297680, -1.5187836, -4.4600635, -1.5691319, -2.2141976, 2.1938362
9: -2.3799126, 0.1479986, -2.3351483, 0.0981154, -2.3388352, 2.3506770

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9622296
time: 5.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9622334
time: 4.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.2939339, -6.2376146, -9.2865896, -6.2992163, -2.2909002, 2.3386960
1: -6.8486962, -4.3246965, -6.8183250, -4.3343172, -2.1426587, 2.1628187
2: -8.8170938, -6.4668517, -8.8025160, -6.4797025, -2.0212293, 2.0258646
3: -10.1717253, -7.5056868, -10.1429958, -7.5239201, -1.9685745, 1.9527068
4: -5.0429010, -2.4692013, -5.0094280, -2.4817460, -2.3797050, 2.3824062
5: -5.4384708, -2.9235749, -5.4285049, -2.9640765, -2.0818925, 2.0975692
6: -13.7766514, -10.6418695, -13.6904173, -10.6508112, -3.0457592, 2.9766264
7: 3.2284083, 5.0548387, 3.2483888, 5.0232892, -1.6792810, 1.6931474
8: -4.5297680, -1.5187836, -4.4855032, -1.5299392, -2.2403574, 2.2033620
9: -2.3799126, 0.1479986, -2.3557863, 0.1167908, -2.3592529, 2.3695979

Time for backsubstitution: 12.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9646690
time: 5.14 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9646711
time: 5.43 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.2614212, -6.2397370, -9.2748222, -6.2840948, -2.2674570, 2.2654524
1: -6.8266397, -4.3344355, -6.8065581, -4.3530550, -2.0778823, 2.0738695
2: -8.7873735, -6.4859028, -8.8043041, -6.4809690, -2.0038500, 1.9928043
3: -10.1268415, -7.5233903, -10.1517401, -7.5329661, -1.9271679, 1.9525503
4: -5.0193801, -2.4906564, -4.9983902, -2.5014248, -2.3214865, 2.3159695
5: -5.3915954, -2.9467700, -5.4418650, -2.9582305, -2.0428653, 2.0657392
6: -13.7599525, -10.6570225, -13.6856966, -10.6719418, -2.9565411, 2.9075961
7: 3.2653928, 5.0516558, 3.2724538, 5.0351205, -1.6558444, 1.6689205
8: -4.5052843, -1.5560017, -4.4984760, -1.5675163, -2.1891823, 2.1930583
9: -2.3621111, 0.1295452, -2.3357840, 0.1208507, -2.3471518, 2.3314097

Time for backsubstitution: 12.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9658181
time: 4.54 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9629411
time: 4.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2614956, -6.2396183, -9.3218060, -6.2319784, -2.2751899, 2.3155894
1: -6.8267608, -4.3343024, -6.8485842, -4.3255572, -2.1453376, 2.1599259
2: -8.7878780, -6.4858317, -8.8266249, -6.4677596, -2.0259013, 2.0359890
3: -10.1272211, -7.5233669, -10.1867476, -7.5047364, -1.9527597, 1.9748715
4: -5.0195045, -2.4906106, -5.0357275, -2.4682763, -2.3677492, 2.3604515
5: -5.3917084, -2.9466376, -5.4743390, -2.9241717, -2.0822477, 2.1072588
6: -13.7600479, -10.6568584, -13.7784414, -10.6070471, -3.0179305, 3.0126462
7: 3.2653751, 5.0516906, 3.2422194, 5.0688491, -1.6809626, 1.6855227
8: -4.5052943, -1.5559268, -4.5504580, -1.5319357, -2.2221818, 2.2322712
9: -2.3622127, 0.1295733, -2.3780177, 0.1574968, -2.3779588, 2.3682008

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9622330, upper bound: 0.9658181
time: 4.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9622330, upper bound: 0.9629411
time: 4.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.2944155, -6.2317185, -9.2992287, -6.2874637, -2.2996292, 2.3209879
1: -6.8491979, -4.3243465, -6.8063784, -4.3399677, -2.1342630, 2.1500378
2: -8.8176308, -6.4634047, -8.7980642, -6.4865313, -2.0346932, 2.0320559
3: -10.1720514, -7.5003362, -10.1351109, -7.5270000, -1.9761875, 1.9642837
4: -5.0447383, -2.4684768, -4.9922571, -2.4915607, -2.3568072, 2.3691535
5: -5.4397144, -2.9162786, -5.4417181, -2.9708004, -2.0956526, 2.1060104
6: -13.7818298, -10.6415148, -13.6900368, -10.6245775, -3.0203104, 2.9452853
7: 3.2261000, 5.0552692, 3.2775807, 5.0363908, -1.6944270, 1.6612920
8: -4.5310321, -1.5154643, -4.4915714, -1.5560660, -2.2286100, 2.2158713
9: -2.3830609, 0.1482711, -2.3482459, 0.1177129, -2.3622041, 2.3666472

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9651065
time: 4.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9622335
time: 4.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.2944155, -6.2317185, -9.3314581, -6.2796717, -2.3047228, 2.3511040
1: -6.8491979, -4.3243465, -6.8286333, -4.3301516, -2.1473808, 2.1740472
2: -8.8176308, -6.4634047, -8.8276758, -6.4648433, -2.0472293, 2.0552862
3: -10.1720514, -7.5003362, -10.1799774, -7.5049758, -1.9885156, 1.9960730
4: -5.0447383, -2.4684768, -5.0174122, -2.4696681, -2.3896914, 2.3945603
5: -5.4397144, -2.9162786, -5.4894118, -2.9410417, -2.1000977, 2.1337867
6: -13.7818298, -10.6415148, -13.7115889, -10.6095734, -3.0552540, 2.9918180
7: 3.2261000, 5.0552692, 3.2389078, 5.0398235, -1.6982031, 1.7014111
8: -4.5310321, -1.5154643, -4.5169878, -1.5168676, -2.2547703, 2.2394354
9: -2.3830609, 0.1482711, -2.3689263, 0.1365010, -2.3827181, 2.3855281

Time for backsubstitution: 12.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 5857

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9651070
time: 5.09 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9647019
time: 5.81 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 23.56 seconds
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9622298
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9622334
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9622308
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9646686
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9658177
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9629434
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9684325
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9596188, upper bound: 0.9655597
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9651059
IS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629407, upper bound: 0.9622335
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9675413
IS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9629410, upper bound: 0.9622334
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9622296
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9622334
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9646690
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9646711
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9658181
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9629411
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9622330, upper bound: 0.9658181
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9622330, upper bound: 0.9629411
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9651065
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655572, upper bound: 0.9622335
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9651070
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 23.56
Output dim: 7, lower bound: -0.9655575, upper bound: 0.9647019

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.2137794, -6.3093643, -9.3217297, -6.2320967, -2.2487564, 2.2797859
1: -6.7853842, -4.3627400, -6.8484616, -4.3256912, -2.0602546, 2.0908625
2: -8.7641144, -6.5075798, -8.8261204, -6.4678278, -1.9668279, 2.0011828
3: -10.0927734, -7.5665693, -10.1863651, -7.5047603, -1.9170783, 1.9297515
4: -4.9767933, -2.5248652, -5.0356026, -2.4683225, -2.3154616, 2.3098543
5: -5.3564520, -3.0019968, -5.4742250, -2.9243045, -2.0479536, 2.0518069
6: -13.6516829, -10.7222109, -13.7783508, -10.6072121, -2.9185591, 2.9543176
7: 3.3016930, 5.0167184, 3.2422385, 5.0688148, -1.6575208, 1.6604846
8: -4.4508424, -1.6002913, -4.5504475, -1.5320072, -2.1729145, 2.1871102
9: -2.3110857, 0.0929170, -2.3779182, 0.1574709, -2.3268023, 2.3331151

Time for backsubstitution: 12.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9651057
time: 4.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9684322
time: 4.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.2460384, -6.3016338, -9.3125372, -6.2807655, -2.2486100, 2.2988384
1: -6.8058748, -4.3528070, -6.8228664, -4.3388181, -2.0814281, 2.0873089
2: -8.7937784, -6.4867759, -8.8236370, -6.4681864, -1.9919686, 2.0120542
3: -10.1359940, -7.5445051, -10.1767521, -7.5146666, -1.9369111, 1.9474425
4: -5.0019689, -2.5036554, -5.0139217, -2.4799314, -2.3322172, 2.3190012
5: -5.4034872, -2.9723353, -5.4777117, -2.9448876, -2.0468340, 2.0759947
6: -13.6732759, -10.7073421, -13.7037802, -10.6358824, -2.9471750, 2.9303803
7: 3.2630086, 5.0201755, 3.2457714, 5.0383968, -1.6672099, 1.6605902
8: -4.4752202, -1.5609097, -4.5122318, -1.5312719, -2.1805425, 2.1887922
9: -2.3315768, 0.1107858, -2.3576019, 0.1337290, -2.3274798, 2.3317547

Time for backsubstitution: 12.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9648511, upper bound: 0.9649133
time: 4.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9648513, upper bound: 0.9675408
time: 4.50 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.2599602, -6.2574968, -9.2748222, -6.2840948, -2.2716255, 2.2476654
1: -6.8251519, -4.3354983, -6.8065581, -4.3530550, -2.0762911, 2.0728555
2: -8.7857609, -6.4963202, -8.8043041, -6.4809690, -1.9878836, 1.9784875
3: -10.1258736, -7.5394979, -10.1517401, -7.5329661, -1.9225206, 1.9351703
4: -5.0139070, -2.4928493, -4.9983902, -2.5014248, -2.3152585, 2.3083396
5: -5.3878374, -2.9687529, -5.4418650, -2.9582305, -2.0449634, 2.0437431
6: -13.7443686, -10.6581078, -13.6856966, -10.6719418, -2.9407263, 2.9120951
7: 3.2723370, 5.0503454, 3.2724538, 5.0351205, -1.6488082, 1.6689433
8: -4.5014663, -1.5659842, -4.4984760, -1.5675163, -2.1798849, 2.1798632
9: -2.3526237, 0.1287555, -2.3357840, 0.1208507, -2.3361602, 2.3263245

Time for backsubstitution: 12.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9624916
time: 4.43 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9658181
time: 4.56 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.3218060, -6.2319784, -2.2793522, 2.2978067
1: -6.8252726, -4.3353658, -6.8485842, -4.3255572, -2.1437411, 2.1588492
2: -8.7862673, -6.4962525, -8.8266249, -6.4677596, -2.0098643, 2.0217278
3: -10.1262522, -7.5394750, -10.1867476, -7.5047364, -1.9481678, 1.9574838
4: -5.0140319, -2.4928031, -5.0357275, -2.4682763, -2.3615003, 2.3547735
5: -5.3879519, -2.9686193, -5.4743390, -2.9241717, -2.0844026, 2.0852542
6: -13.7444639, -10.6579409, -13.7784414, -10.6070471, -3.0021148, 3.0145078
7: 3.2723179, 5.0503802, 3.2422194, 5.0688491, -1.6739264, 1.6855459
8: -4.5014772, -1.5659080, -4.5504580, -1.5319357, -2.2128816, 2.2190757
9: -2.3527238, 0.1287827, -2.3780177, 0.1574968, -2.3669758, 2.3632207

Time for backsubstitution: 12.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6181
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 6181

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9622329, upper bound: 0.9624916
time: 4.44 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9622330, upper bound: 0.9658181
time: 4.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 21.89 seconds
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9651057
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9596167, upper bound: 0.9684322
IS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9648511, upper bound: 0.9649133
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9648513, upper bound: 0.9675408
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9624916
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9622328, upper bound: 0.9658181
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9622329, upper bound: 0.9624916
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 21.89
Output dim: 7, lower bound: -0.9622330, upper bound: 0.9658181

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2137794, -6.3093643, -9.3377533, -6.2300525, -2.2510319, 2.2901835
1: -6.7853842, -4.3627400, -6.8578677, -4.3213959, -2.0647364, 2.0979269
2: -8.7641144, -6.5075798, -8.8407087, -6.4588904, -1.9732819, 2.0131471
3: -10.0927734, -7.5665693, -10.2075691, -7.4975677, -1.9235127, 1.9405993
4: -4.9767933, -2.5248652, -5.0470524, -2.4586804, -2.3223171, 2.3147941
5: -5.3564520, -3.0019968, -5.4967060, -2.9153676, -2.0503769, 2.0572529
6: -13.6516829, -10.7222109, -13.7873125, -10.6015205, -2.9174185, 2.9589667
7: 3.3016930, 5.0167184, 3.2235737, 5.0704103, -1.6589606, 1.6802063
8: -4.4508424, -1.6002913, -4.5584726, -1.5124140, -2.1825438, 2.1865482
9: -2.3110857, 0.0929170, -2.3867459, 0.1670849, -2.3334932, 2.3414054

Time for backsubstitution: 12.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9484799, upper bound: 0.9680024
time: 4.52 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9591877, upper bound: 0.9680031
time: 5.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2460384, -6.3016338, -9.3377533, -6.2300525, -2.2813950, 2.3055267
1: -6.8058748, -4.3528070, -6.8578677, -4.3213959, -2.0863619, 2.1111336
2: -8.7937784, -6.4867759, -8.8407087, -6.4588904, -1.9965954, 2.0276020
3: -10.1359940, -7.5445051, -10.2075691, -7.4975677, -1.9535389, 1.9663656
4: -5.0019689, -2.5036554, -5.0470524, -2.4586804, -2.3465319, 2.3458419
5: -5.4034872, -2.9723353, -5.4967060, -2.9153676, -2.0693464, 2.0826294
6: -13.6732759, -10.7073421, -13.7873125, -10.6015205, -2.9588766, 2.9937558
7: 3.2630086, 5.0201755, 3.2235737, 5.0704103, -1.6996019, 1.6839867
8: -4.4752202, -1.5609097, -4.5584726, -1.5124140, -2.1995831, 2.2128253
9: -2.3315768, 0.1107858, -2.3867459, 0.1670849, -2.3594408, 2.3607574

Time for backsubstitution: 12.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9536458, upper bound: 0.9671943
time: 5.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9644693, upper bound: 0.9671937
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.2599602, -6.2574968, -9.2908669, -6.2820559, -2.2759805, 2.2582202
1: -6.8251519, -4.3354983, -6.8161855, -4.3487291, -2.0807590, 2.0802240
2: -8.7857609, -6.4963202, -8.8189554, -6.4719419, -1.9944129, 1.9930224
3: -10.1258736, -7.5394979, -10.1729650, -7.5255885, -1.9289560, 1.9462218
4: -5.0139070, -2.4928493, -5.0099001, -2.4916728, -2.3206010, 2.3207021
5: -5.3878374, -2.9687529, -5.4643111, -2.9493253, -2.0515308, 2.0491114
6: -13.7443686, -10.6581078, -13.6946964, -10.6661882, -2.9395676, 2.9216695
7: 3.2723370, 5.0503454, 3.2535410, 5.0367002, -1.6504142, 1.6889505
8: -4.5014663, -1.5659842, -4.5066185, -1.5478115, -2.1867971, 2.1792600
9: -2.3526237, 0.1287555, -2.3447146, 0.1304823, -2.3474569, 2.3347020

Time for backsubstitution: 12.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9511469, upper bound: 0.9653899
time: 5.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9617984, upper bound: 0.9653888
time: 5.21 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.2600327, -6.2573767, -9.3378305, -6.2299337, -2.2837143, 2.3082204
1: -6.8252726, -4.3353658, -6.8579874, -4.3212638, -2.1483521, 2.1662138
2: -8.7862673, -6.4962525, -8.8412151, -6.4588223, -2.0163307, 2.0300937
3: -10.1262522, -7.5394750, -10.2079506, -7.4975433, -1.9545991, 1.9683311
4: -5.0140319, -2.4928031, -5.0471773, -2.4586329, -2.3667231, 2.3597286
5: -5.3879519, -2.9686193, -5.4968190, -2.9152360, -2.0909028, 2.0906541
6: -13.7444639, -10.6579409, -13.7874041, -10.6013517, -3.0009303, 3.0191574
7: 3.2723179, 5.0503802, 3.2235560, 5.0704441, -1.6755619, 1.7052686
8: -4.5014772, -1.5659080, -4.5584836, -1.5123401, -2.2197466, 2.2185130
9: -2.3527238, 0.1287827, -2.3868451, 0.1671113, -2.3736672, 2.3714547

Time for backsubstitution: 12.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 874
type: A, layer: 1, pos: 5751
type: A, layer: 1, pos: 6235
type: A, layer: 1, pos: 4627
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 146

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 874

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9511472, upper bound: 0.9653899
time: 5.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.9617986, upper bound: 0.9653885
time: 4.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 22.35 seconds
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9484799, upper bound: 0.9680024
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9591877, upper bound: 0.9680031
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9536458, upper bound: 0.9671943
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9644693, upper bound: 0.9671937
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9511469, upper bound: 0.9653899
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9617984, upper bound: 0.9653888
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9511472, upper bound: 0.9653899
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 22.35
Output dim: 7, lower bound: -0.9617986, upper bound: 0.9653885

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.2110291, -6.3135376, -9.3369370, -6.2335601, -2.2396350, 2.2785532
1: -6.7814722, -4.3754048, -6.8568830, -4.3260121, -2.0727563, 2.0923626
2: -8.7581949, -6.5096736, -8.8377104, -6.4593587, -1.9710774, 2.0058322
3: -10.0799227, -7.5689025, -10.2027645, -7.4979906, -1.9176593, 1.9281141
4: -4.9723854, -2.5410151, -5.0462966, -2.4654160, -2.3189974, 2.2980208
5: -5.3547187, -3.0033121, -5.4961247, -2.9167848, -2.0466990, 2.0549688
6: -13.6509447, -10.7277412, -13.7863979, -10.6037951, -2.9104929, 2.9518380
7: 3.3143868, 5.0139737, 3.2287016, 5.0701609, -1.6456227, 1.6720052
8: -4.4454708, -1.6195765, -4.5575733, -1.5195279, -2.1629758, 2.1661849
9: -2.3063855, 0.0783790, -2.3858252, 0.1610305, -2.3161335, 2.3256021

Time for backsubstitution: 12.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5751
type: B, layer: 1, pos: 874
type: B, layer: 1, pos: 6235
type: B, layer: 1, pos: 4627
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 146

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5751

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9474167, upper bound: 0.9679918
time: 4.91 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.9484651, upper bound: 0.9679950
time: 4.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.2136583, -6.3093853, -9.3377533, -6.2300525, -2.2490811, 2.2814548
1: -6.7853694, -4.3627515, -6.8578677, -4.3213959, -2.0647178, 2.1153998
2: -8.7640305, -6.5075932, -8.8407087, -6.4588904, -1.9732337, 2.0122235
3: -10.0927677, -7.5666418, -10.2075691, -7.4975677, -1.9295740, 1.9383651
4: -4.9767556, -2.5248923, -5.0470524, -2.4586804, -2.3222847, 2.3025005
5: -5.3564177, -3.0020514, -5.4967060, -2.9153676, -2.0502586, 2.0568271
6: -13.6516094, -10.7222166, -13.7873125, -10.6015205, -2.9170465, 2.9575472
7: 3.3016992, 5.0166917, 3.2235737, 5.0704103, -1.6494119, 1.6801775
8: -4.4507971, -1.6003475, -4.5584726, -1.5124140, -2.1794434, 2.1786711
9: -2.3110108, 0.0928997, -2.3867459, 0.1670849, -2.3308058, 2.3290730

Time for backsubstitution: 12.46 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=4, k_high=4, k_mid=4, eps_mid=0.0156250, abs_max=1.6687421798706055
rel_dist={7: [-0.9703767544746134, 0.9703745834961706]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2423.48 seconds
