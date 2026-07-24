## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2745331572


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.6024015, 0.6024015)
1: (-7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5952206, 0.5952203)
2: (-2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5117292, 0.5117292)
3: (5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5401940, 0.5401940)
4: (-11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5841854, 0.5841851)
5: (-2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5143909, 0.5143909)
6: (-9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6629021, 0.6629021)
7: (-7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6346526, 0.6346531)
8: (-2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5288428, 0.5288427)
9: (-4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4562324, 0.4562323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.17 + 34.14 = 57.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2756352, upper bound: 0.2756360

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5847
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756316, upper bound: 0.2742700
time: 3.65 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756342, upper bound: 0.2756348
time: 3.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.78 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.78
Output dim: 3, lower bound: -0.2756316, upper bound: 0.2742700
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.78
Output dim: 3, lower bound: -0.2756342, upper bound: 0.2756348

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.9413338, -7.8348966, -8.9453316, -7.8348489, -0.5942743, 0.5982687
1: -7.1683588, -6.1090517, -7.1684322, -6.1022072, -0.5883071, 0.5815539
2: -2.9238944, -1.8808579, -2.9240360, -1.8793659, -0.5096104, 0.5079247
3: 5.8406715, 6.8146114, 5.8405967, 6.8165483, -0.5379941, 0.5359386
4: -11.6826086, -10.4385166, -11.6845350, -10.4384327, -0.5801404, 0.5820560
5: -2.0643668, -1.1130342, -2.0644321, -1.1113875, -0.5122979, 0.5103670
6: -9.6052046, -8.3941240, -9.6056538, -8.3930225, -0.6604393, 0.6593926
7: -7.1135182, -6.0869141, -7.1177268, -6.0867782, -0.6260538, 0.6301866
8: -2.1370311, -1.2095141, -2.1370578, -1.2086983, -0.5268614, 0.5258393
9: -4.3103094, -3.3545299, -4.3138261, -3.3544011, -0.4491314, 0.4525255

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 933

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742700
time: 4.72 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742707
time: 3.77 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.9516983, -7.8254080, -8.9493093, -7.8348122, -0.6007271, 0.6117704
1: -7.1853533, -6.0936871, -7.1685009, -6.0953608, -0.6043847, 0.5885172
2: -2.9299157, -1.8766956, -2.9241748, -1.8778169, -0.5175109, 0.5118265
3: 5.8356786, 6.8186660, 5.8405275, 6.8184690, -0.5450864, 0.5377790
4: -11.6876106, -10.4337063, -11.6864643, -10.4383497, -0.5838010, 0.5890362
5: -2.0693042, -1.1089401, -2.0644875, -1.1097522, -0.5188926, 0.5131853
6: -9.6117916, -8.3904877, -9.6060905, -8.3918839, -0.6686599, 0.6639762
7: -7.1238627, -6.0762811, -7.1219311, -6.0866461, -0.6323361, 0.6402613
8: -2.1395049, -1.2074661, -2.1370850, -1.2078900, -0.5305883, 0.5287169
9: -4.3174591, -3.3465655, -4.3173137, -3.3542793, -0.4527261, 0.4641131

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 933

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2756323
time: 4.03 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2756340
time: 6.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.74 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.74
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742700
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 31.74
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742707
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.74
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2756323
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.74
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2756340

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.9516993, -7.8254576, -8.9413338, -7.8348966, -0.6047349, 0.6036410
1: -7.1853533, -6.0937839, -7.1683588, -6.1090517, -0.5907631, 0.5959451
2: -2.9299145, -1.8767805, -2.9238944, -1.8808579, -0.5138466, 0.5117061
3: 5.8357153, 6.8186660, 5.8406715, 6.8146114, -0.5408475, 0.5398608
4: -11.6876106, -10.4337063, -11.6826086, -10.4385166, -0.5852580, 0.5850599
5: -2.0692427, -1.1089402, -2.0643668, -1.1130342, -0.5148674, 0.5142725
6: -9.6117916, -8.3905163, -9.6052046, -8.3941240, -0.6656220, 0.6626439
7: -7.1238537, -6.0762820, -7.1135182, -6.0869141, -0.6364622, 0.6318278
8: -2.1394591, -1.2074666, -2.1370311, -1.2095141, -0.5278786, 0.5275737
9: -4.3174605, -3.3465898, -4.3103094, -3.3545299, -0.4561949, 0.4570957

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5842
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 5842
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 933

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 928

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740087, upper bound: 0.2754885
time: 3.63 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742175, upper bound: 0.2755795
time: 3.68 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.9516983, -7.8254080, -8.9516983, -7.8254080, -0.6037192, 0.6037190
1: -7.1853533, -6.0936871, -7.1853533, -6.0936871, -0.5925112, 0.5925112
2: -2.9299157, -1.8766956, -2.9299157, -1.8766956, -0.5173149, 0.5173150
3: 5.8356786, 6.8186660, 5.8356786, 6.8186660, -0.5384645, 0.5384645
4: -11.6876106, -10.4337063, -11.6876106, -10.4337063, -0.5861804, 0.5861802
5: -2.0693042, -1.1089401, -2.0693042, -1.1089401, -0.5143526, 0.5143526
6: -9.6117916, -8.3904877, -9.6117916, -8.3904877, -0.6667273, 0.6667273
7: -7.1238627, -6.0762811, -7.1238627, -6.0762811, -0.6360354, 0.6360354
8: -2.1395049, -1.2074661, -2.1395049, -1.2074661, -0.5289481, 0.5289482
9: -4.3174591, -3.3465655, -4.3174591, -3.3465655, -0.4548784, 0.4548784

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 5842
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741266, upper bound: 0.2753727
time: 3.58 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742175, upper bound: 0.2753222
time: 7.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.21 seconds
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 3, lower bound: -0.2740087, upper bound: 0.2754885
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 3, lower bound: -0.2742175, upper bound: 0.2755795
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 3, lower bound: -0.2741266, upper bound: 0.2753727
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.21
Output dim: 3, lower bound: -0.2742175, upper bound: 0.2753222

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.9494267, -7.8256254, -8.9366398, -7.8352480, -0.6020944, 0.5986211
1: -7.1849222, -6.0950804, -7.1674595, -6.1115346, -0.5872471, 0.5927052
2: -2.9297264, -1.8768377, -2.9235036, -1.8809774, -0.5132184, 0.5106986
3: 5.8358994, 6.8179755, 5.8410521, 6.8131847, -0.5383141, 0.5373780
4: -11.6862946, -10.4338322, -11.6800947, -10.4387760, -0.5822144, 0.5809891
5: -2.0691392, -1.1100006, -2.0641513, -1.1152298, -0.5125420, 0.5129738
6: -9.6085491, -8.3905659, -9.5984955, -8.3942289, -0.6620853, 0.6554801
7: -7.1229467, -6.0763788, -7.1116390, -6.0871100, -0.6343045, 0.6285869
8: -2.1339521, -1.2076354, -2.1256275, -1.2098594, -0.5219606, 0.5159514
9: -4.3173103, -3.3481297, -4.3099999, -3.3577151, -0.4527344, 0.4552635

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 933

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2738025, upper bound: 0.2754869
time: 6.28 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740079, upper bound: 0.2754878
time: 3.62 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -8.9516830, -7.8254561, -8.9415894, -7.8296785, -0.6100938, 0.6018434
1: -7.1853528, -6.0937877, -7.1709709, -6.1090527, -0.5892678, 0.5964669
2: -2.9299152, -1.8779988, -2.9224665, -1.8825893, -0.5151761, 0.5088660
3: 5.8357134, 6.8186626, 5.8405018, 6.8146782, -0.5401134, 0.5386668
4: -11.6875944, -10.4337063, -11.6826811, -10.4358759, -0.5870278, 0.5843589
5: -2.0692434, -1.1089420, -2.0672669, -1.1128824, -0.5141088, 0.5167722
6: -9.6117859, -8.3905163, -9.6062527, -8.3861923, -0.6693832, 0.6597555
7: -7.1238508, -6.0762844, -7.1140947, -6.0857310, -0.6368568, 0.6317214
8: -2.1394520, -1.2074714, -2.1370287, -1.1950045, -0.5339701, 0.5207889
9: -4.3174500, -3.3465929, -4.3136835, -3.3541369, -0.4553599, 0.4597398

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 5842
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740111, upper bound: 0.2755779
time: 5.77 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742167, upper bound: 0.2755787
time: 3.81 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.9470043, -7.8257580, -8.9494276, -7.8255768, -0.5987017, 0.6010780
1: -7.1844530, -6.0961699, -7.1849203, -6.0949831, -0.5895646, 0.5889902
2: -2.9295228, -1.8768148, -2.9297266, -1.8767531, -0.5163074, 0.5166874
3: 5.8360643, 6.8172393, 5.8358631, 6.8179755, -0.5359812, 0.5359311
4: -11.6850986, -10.4339657, -11.6862955, -10.4338322, -0.5821092, 0.5831354
5: -2.0690897, -1.1111355, -2.0692005, -1.1100007, -0.5130537, 0.5120325
6: -9.6050816, -8.3905916, -9.6085520, -8.3905373, -0.6595631, 0.6631899
7: -7.1219864, -6.0764790, -7.1229558, -6.0763788, -0.6334138, 0.6343877
8: -2.1281009, -1.2078104, -2.1339965, -1.2076354, -0.5173249, 0.5230299
9: -4.3171496, -3.3497522, -4.3173094, -3.3481057, -0.4530463, 0.4514174

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 5842
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 5805

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2751690
time: 4.19 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2753728
time: 3.72 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9519539, -7.8201885, -8.9516840, -7.8254085, -0.6019232, 0.6090779
1: -7.1879673, -6.0936880, -7.1853528, -6.0936904, -0.5937829, 0.5910153
2: -2.9284859, -1.8784285, -2.9299147, -1.8779140, -0.5144753, 0.5186450
3: 5.8355136, 6.8187332, 5.8356791, 6.8186626, -0.5372708, 0.5377303
4: -11.6876860, -10.4310656, -11.6875954, -10.4337091, -0.5854812, 0.5879495
5: -2.0722046, -1.1087867, -2.0693045, -1.1089420, -0.5173481, 0.5135663
6: -9.6128387, -8.3825569, -9.6117859, -8.3904867, -0.6638401, 0.6723452
7: -7.1244411, -6.0750971, -7.1238599, -6.0762844, -0.6355696, 0.6382396
8: -2.1395020, -1.1929555, -2.1394973, -1.2074709, -0.5221635, 0.5360628
9: -4.3208342, -3.3461747, -4.3174500, -3.3465686, -0.4583632, 0.4540434

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742182, upper bound: 0.2753789
time: 3.83 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742182, upper bound: 0.2755817
time: 3.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.84 seconds
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2738025, upper bound: 0.2754869
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2740079, upper bound: 0.2754878
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2740111, upper bound: 0.2755779
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2742167, upper bound: 0.2755787
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2751690
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2753728
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2742182, upper bound: 0.2753789
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.84
Output dim: 3, lower bound: -0.2742182, upper bound: 0.2755817

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -8.9482279, -7.8256388, -8.9341402, -7.8352766, -0.6001754, 0.5953751
1: -7.1827412, -6.0957499, -7.1630497, -6.1129932, -0.5825596, 0.5871811
2: -2.9283464, -1.8770571, -2.9207852, -1.8814225, -0.5110602, 0.5071408
3: 5.8360167, 6.8174720, 5.8412976, 6.8121843, -0.5370746, 0.5366205
4: -11.6839828, -10.4341097, -11.6754122, -10.4393644, -0.5792527, 0.5755966
5: -2.0688548, -1.1102760, -2.0635858, -1.1158407, -0.5106597, 0.5112642
6: -9.6080055, -8.3908234, -9.5973845, -8.3947105, -0.6584303, 0.6517329
7: -7.1227093, -6.0765066, -7.1111164, -6.0873718, -0.6332312, 0.6273142
8: -2.1321716, -1.2077670, -2.1220355, -1.2101440, -0.5197304, 0.5118959
9: -4.3169103, -3.3484082, -4.3091927, -3.3582606, -0.4517355, 0.4540253

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5805

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2738006, upper bound: 0.2753164
time: 3.57 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2738006, upper bound: 0.2753156
time: 3.81 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -8.9494247, -7.8256235, -8.9380417, -7.8343887, -0.6031718, 0.5997570
1: -7.1849155, -6.0950823, -7.1675901, -6.1034455, -0.5890094, 0.5898708
2: -2.9297228, -1.8768377, -2.9247382, -1.8791151, -0.5150461, 0.5115139
3: 5.8359003, 6.8179750, 5.8398027, 6.8135481, -0.5384386, 0.5386451
4: -11.6862860, -10.4338331, -11.6807117, -10.4322968, -0.5890179, 0.5795410
5: -2.0691385, -1.1100016, -2.0643377, -1.1145141, -0.5121543, 0.5133653
6: -9.6085491, -8.3905659, -9.5997992, -8.3929710, -0.6642766, 0.6553919
7: -7.1229467, -6.0763779, -7.1126375, -6.0865955, -0.6344624, 0.6290684
8: -2.1339464, -1.2076354, -2.1261826, -1.2052851, -0.5255216, 0.5145667
9: -4.3173075, -3.3481307, -4.3104553, -3.3568187, -0.4535369, 0.4554470

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5842
type: A, layer: 1, pos: 5805

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740060, upper bound: 0.2753164
time: 3.91 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740060, upper bound: 0.2753156
time: 3.85 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -8.9504852, -7.8254728, -8.9390955, -7.8297062, -0.6081746, 0.5985980
1: -7.1831741, -6.0944591, -7.1665587, -6.1105103, -0.5845809, 0.5909410
2: -2.9285345, -1.8782184, -2.9197502, -1.8830345, -0.5130172, 0.5053091
3: 5.8358326, 6.8181601, 5.8407483, 6.8136768, -0.5388739, 0.5379096
4: -11.6852827, -10.4339867, -11.6779966, -10.4364681, -0.5840659, 0.5789745
5: -2.0689585, -1.1092174, -2.0666990, -1.1134920, -0.5122262, 0.5149549
6: -9.6112404, -8.3907766, -9.6051407, -8.3866768, -0.6655618, 0.6560092
7: -7.1236134, -6.0764117, -7.1135736, -6.0859928, -0.6357832, 0.6304531
8: -2.1376696, -1.2076039, -2.1334367, -1.1952901, -0.5311310, 0.5167333
9: -4.3170505, -3.3468716, -4.3128781, -3.3546839, -0.4543616, 0.4584807

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 5842
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740097, upper bound: 0.2753693
time: 4.76 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740097, upper bound: 0.2755779
time: 6.21 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -8.9516792, -7.8254561, -8.9429951, -7.8288174, -0.6104556, 0.6029799
1: -7.1853476, -6.0937891, -7.1711044, -6.1009636, -0.5910305, 0.5936335
2: -2.9299097, -1.8779991, -2.9237015, -1.8807268, -0.5170045, 0.5096815
3: 5.8357143, 6.8186631, 5.8392544, 6.8150420, -0.5402372, 0.5399346
4: -11.6875820, -10.4337063, -11.6832943, -10.4293976, -0.5938318, 0.5829227
5: -2.0692425, -1.1089429, -2.0674522, -1.1121657, -0.5137210, 0.5163903
6: -9.6117821, -8.3905163, -9.6075611, -8.3849354, -0.6692665, 0.6596708
7: -7.1238513, -6.0762835, -7.1150923, -6.0852141, -0.6370139, 0.6322069
8: -2.1394453, -1.2074728, -2.1375842, -1.1904287, -0.5343587, 0.5194041
9: -4.3174486, -3.3465931, -4.3141398, -3.3532419, -0.4561627, 0.4597692

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5842
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 5842

## Relational analysis of NS_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753188
time: 3.81 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753156
time: 3.92 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -8.9446049, -7.8257856, -8.9482250, -7.8255873, -0.5955529, 0.5991592
1: -7.1800528, -6.0975957, -7.1827426, -6.0956535, -0.5841169, 0.5851765
2: -2.9267797, -1.8772650, -2.9283471, -1.8769732, -0.5126863, 0.5145271
3: 5.8363008, 6.8162384, 5.8359823, 6.8174720, -0.5352252, 0.5346911
4: -11.6804209, -10.4345570, -11.6839828, -10.4341097, -0.5767176, 0.5801752
5: -2.0685241, -1.1116996, -2.0689163, -1.1102759, -0.5113232, 0.5102907
6: -9.6039648, -8.3910751, -9.6080074, -8.3907976, -0.6558018, 0.6595361
7: -7.1214933, -6.0767398, -7.1227183, -6.0765066, -0.6320715, 0.6333394
8: -2.1245112, -1.2080803, -2.1322165, -1.2077684, -0.5132517, 0.5207775
9: -4.3163443, -3.3503249, -4.3169103, -3.3483837, -0.4518074, 0.4503725

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5842
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751670
time: 3.96 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751673
time: 3.47 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -8.9483786, -7.8248963, -8.9494247, -7.8255758, -0.5998998, 0.6021564
1: -7.1845870, -6.0880928, -7.1849155, -6.0949845, -0.5871596, 0.5959520
2: -2.9307632, -1.8749578, -2.9297223, -1.8767543, -0.5170739, 0.5185170
3: 5.8348098, 6.8176050, 5.8358650, 6.8179750, -0.5372493, 0.5360546
4: -11.6857233, -10.4274807, -11.6862869, -10.4338331, -0.5806653, 0.5899417
5: -2.0692751, -1.1104202, -2.0692005, -1.1100016, -0.5134459, 0.5121198
6: -9.6064100, -8.3893347, -9.6085491, -8.3905382, -0.6594863, 0.6653826
7: -7.1229792, -6.0759449, -7.1229563, -6.0763779, -0.6344848, 0.6347885
8: -2.1286573, -1.2032342, -2.1339908, -1.2076354, -0.5159390, 0.5275784
9: -4.3176064, -3.3488703, -4.3173075, -3.3481052, -0.4532337, 0.4521847

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 5842
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 933

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753709
time: 4.08 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753720
time: 3.78 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -8.9495583, -7.8202171, -8.9504833, -7.8254242, -0.5987747, 0.6071594
1: -7.1835613, -6.0951138, -7.1831741, -6.0943613, -0.5883274, 0.5872016
2: -2.9257441, -1.8788762, -2.9285345, -1.8781333, -0.5108535, 0.5164850
3: 5.8357506, 6.8177319, 5.8357978, 6.8181601, -0.5365143, 0.5364907
4: -11.6830072, -10.4316587, -11.6852827, -10.4339857, -0.5800967, 0.5849886
5: -2.0716388, -1.1093504, -2.0690203, -1.1092174, -0.5156174, 0.5118244
6: -9.6117220, -8.3830385, -9.6112394, -8.3907490, -0.6600792, 0.6685232
7: -7.1239500, -6.0753584, -7.1236229, -6.0764117, -0.6342273, 0.6371918
8: -2.1359110, -1.1932263, -2.1377144, -1.2076039, -0.5180898, 0.5332201
9: -4.3200288, -3.3467486, -4.3170505, -3.3468473, -0.4571241, 0.4529984

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740146, upper bound: 0.2753773
time: 3.97 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740146, upper bound: 0.2753767
time: 4.05 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -8.9533329, -7.8193264, -8.9516792, -7.8254080, -0.6031215, 0.6101565
1: -7.1881008, -6.0856113, -7.1853476, -6.0936923, -0.5913777, 0.5979729
2: -2.9297283, -1.8765693, -2.9299092, -1.8779137, -0.5152411, 0.5204755
3: 5.8342600, 6.8190975, 5.8356800, 6.8186631, -0.5385394, 0.5378537
4: -11.6883068, -10.4245825, -11.6875858, -10.4337063, -0.5840471, 0.5947552
5: -2.0723915, -1.1080728, -2.0693035, -1.1089429, -0.5177405, 0.5136540
6: -9.6141720, -8.3812990, -9.6117840, -8.3904886, -0.6637664, 0.6722271
7: -7.1254349, -6.0745654, -7.1238594, -6.0762835, -0.6366410, 0.6386406
8: -2.1400576, -1.1883793, -2.1394906, -1.2074723, -0.5207772, 0.5364586
9: -4.3212910, -3.3452935, -4.3174486, -3.3465693, -0.4585506, 0.4548110

Time for backsubstitution: 22.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 5842

## Relational analysis of NS_A2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739589, upper bound: 0.2752855
time: 5.65 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739589, upper bound: 0.2754903
time: 3.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 38.73 seconds
NS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2738006, upper bound: 0.2753164
NS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2738006, upper bound: 0.2753156
NS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2740060, upper bound: 0.2753164
NS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2740060, upper bound: 0.2753156
NS_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2740097, upper bound: 0.2753693
NS_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2740097, upper bound: 0.2755779
NS_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753188
NS_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753156
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751670
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751673
NS_A2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753709
NS_A2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753720
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2740146, upper bound: 0.2753773
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2740146, upper bound: 0.2753767
NS_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739589, upper bound: 0.2752855
NS_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 38.73
Output dim: 3, lower bound: -0.2739589, upper bound: 0.2754903

## BFS NS instance: NS_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -8.9458027, -7.8258219, -8.9341402, -7.8352766, -0.5976529, 0.5952291
1: -7.1822734, -6.0969377, -7.1630497, -6.1129932, -0.5819833, 0.5863276
2: -2.9281423, -1.8771191, -2.9207852, -1.8814225, -0.5106091, 0.5070696
3: 5.8362141, 6.8167362, 5.8412976, 6.8121843, -0.5361116, 0.5356073
4: -11.6827841, -10.4342461, -11.6754122, -10.4393644, -0.5778897, 0.5752594
5: -2.0687439, -1.1114092, -2.0635858, -1.1158407, -0.5105313, 0.5101252
6: -9.6045389, -8.3908787, -9.5973845, -8.3947105, -0.6547241, 0.6516545
7: -7.1217389, -6.0766068, -7.1111164, -6.0873718, -0.6324100, 0.6272761
8: -2.1262755, -1.2079425, -2.1220355, -1.2101440, -0.5138003, 0.5116715
9: -4.3167515, -3.3500557, -4.3091927, -3.3582606, -0.4516298, 0.4522907

Time for backsubstitution: 22.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 5842
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 5842
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 5805

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2738006, upper bound: 0.2751078
time: 3.84 seconds

## Relational analysis of NS_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2738006, upper bound: 0.2753164
time: 4.03 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.31 + 546.49 = 603.80 seconds
