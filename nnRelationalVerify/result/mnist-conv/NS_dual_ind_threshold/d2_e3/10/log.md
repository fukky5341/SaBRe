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
execution time: IAR + RelationalAnalysis = 22.90 + 33.47 = 56.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2756352, upper bound: 0.2756360

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5847
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5847

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756316, upper bound: 0.2742700
time: 3.41 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756342, upper bound: 0.2756348
time: 3.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.18 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.18
Output dim: 3, lower bound: -0.2756316, upper bound: 0.2742700
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.18
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

Time for backsubstitution: 21.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742700
time: 4.78 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742707
time: 3.81 seconds

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

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5847
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5847

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2756323
time: 4.04 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2756340
time: 6.41 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.95 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.95
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742700
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 31.95
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2742707
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.95
Output dim: 3, lower bound: -0.2742700, upper bound: 0.2756323
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.95
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

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2740636
time: 8.21 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2756312
time: 3.88 seconds

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

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2754314
time: 3.76 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2756315
time: 3.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.87 seconds
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 29.87
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2740636
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2756312
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2754314
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.87
Output dim: 3, lower bound: -0.2742693, upper bound: 0.2756315

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9530745, -7.8245955, -8.9413319, -7.8348961, -0.6059330, 0.6037705
1: -7.1854877, -6.0857086, -7.1683531, -6.1090541, -0.5879284, 0.5977764
2: -2.9311569, -1.8749235, -2.9238913, -1.8808582, -0.5146129, 0.5135357
3: 5.8344617, 6.8190308, 5.8406706, 6.8146100, -0.5418843, 0.5399835
4: -11.6882343, -10.4272232, -11.6825962, -10.4385204, -0.5838127, 0.5918658
5: -2.0694304, -1.1082252, -2.0643668, -1.1130354, -0.5144855, 0.5143639
6: -9.6131153, -8.3892593, -9.6052017, -8.3941240, -0.6655450, 0.6648328
7: -7.1248493, -6.0757504, -7.1135187, -6.0869155, -0.6373072, 0.6319870
8: -2.1400151, -1.2028909, -2.1370249, -1.2095141, -0.5264981, 0.5321355
9: -4.3179169, -3.3457093, -4.3103085, -3.3545296, -0.4563811, 0.4578640

Time for backsubstitution: 21.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 928

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740079, upper bound: 0.2754871
time: 4.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742167, upper bound: 0.2755787
time: 3.79 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.9493008, -7.8254356, -8.9504995, -7.8254204, -0.6005697, 0.6018002
1: -7.1809516, -6.0951118, -7.1831732, -6.0943556, -0.5870640, 0.5886977
2: -2.9271719, -1.8771443, -2.9285352, -1.8769147, -0.5136938, 0.5151547
3: 5.8359184, 6.8176637, 5.8357973, 6.8181610, -0.5377083, 0.5372250
4: -11.6829338, -10.4342985, -11.6852999, -10.4339867, -0.5807886, 0.5832212
5: -2.0687392, -1.1095047, -2.0690198, -1.1092135, -0.5126218, 0.5126107
6: -9.6106730, -8.3909702, -9.6112461, -8.3907461, -0.6629658, 0.6630728
7: -7.1233706, -6.0765419, -7.1236248, -6.0764098, -0.6346936, 0.6349878
8: -2.1359148, -1.2077370, -2.1377211, -1.2075987, -0.5248746, 0.5266953
9: -4.3166537, -3.3471389, -4.3170600, -3.3468454, -0.4536395, 0.4538333

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740093, upper bound: 0.2752879
time: 4.28 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742181, upper bound: 0.2753789
time: 3.82 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9530754, -7.8245482, -8.9516935, -7.8254080, -0.6049173, 0.6047974
1: -7.1854877, -6.0856109, -7.1853476, -6.0936875, -0.5901067, 0.5994681
2: -2.9311571, -1.8748384, -2.9299102, -1.8766952, -0.5180814, 0.5191444
3: 5.8344259, 6.8190308, 5.8356791, 6.8186646, -0.5397327, 0.5385882
4: -11.6882334, -10.4272242, -11.6876030, -10.4337091, -0.5847344, 0.5929856
5: -2.0694911, -1.1082243, -2.0693047, -1.1089408, -0.5147449, 0.5144402
6: -9.6131172, -8.3892307, -9.6117897, -8.3904867, -0.6666498, 0.6689193
7: -7.1248579, -6.0757504, -7.1238632, -6.0762825, -0.6371067, 0.6364367
8: -2.1400604, -1.2028894, -2.1394982, -1.2074661, -0.5275623, 0.5334964
9: -4.3179164, -3.3456852, -4.3174586, -3.3465672, -0.4550661, 0.4556459

Time for backsubstitution: 22.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2740093, upper bound: 0.2754908
time: 3.99 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742181, upper bound: 0.2755809
time: 4.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.79 seconds
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.79
Output dim: 3, lower bound: -0.2740079, upper bound: 0.2754871
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.79
Output dim: 3, lower bound: -0.2742167, upper bound: 0.2755787
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.79
Output dim: 3, lower bound: -0.2740093, upper bound: 0.2752879
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.79
Output dim: 3, lower bound: -0.2742181, upper bound: 0.2753789
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.79
Output dim: 3, lower bound: -0.2740093, upper bound: 0.2754908
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.79
Output dim: 3, lower bound: -0.2742181, upper bound: 0.2755809

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.9508038, -7.8247643, -8.9366350, -7.8352470, -0.6032920, 0.5987351
1: -7.1850562, -6.0870042, -7.1674533, -6.1115355, -0.5844128, 0.5945368
2: -2.9309688, -1.8749809, -2.9234977, -1.8809774, -0.5139852, 0.5125285
3: 5.8346477, 6.8183413, 5.8410540, 6.8131843, -0.5391320, 0.5375009
4: -11.6869173, -10.4273472, -11.6800861, -10.4387760, -0.5807681, 0.5877953
5: -2.0693262, -1.1092863, -2.0641518, -1.1152296, -0.5121598, 0.5130649
6: -9.6098785, -8.3893089, -9.5984945, -8.3942289, -0.6620095, 0.6576686
7: -7.1239409, -6.0758457, -7.1116385, -6.0871119, -0.6348500, 0.6287464
8: -2.1345086, -1.2030592, -2.1256227, -1.2098589, -0.5205808, 0.5205128
9: -4.3177667, -3.3472474, -4.3099985, -3.3577156, -0.4529203, 0.4560317

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753164
time: 3.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753164
time: 3.99 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -8.9530602, -7.8245974, -8.9415855, -7.8296785, -0.6112919, 0.6019764
1: -7.1854863, -6.0857115, -7.1709657, -6.1090546, -0.5864331, 0.5982985
2: -2.9311562, -1.8761411, -2.9224625, -1.8825898, -0.5159419, 0.5106950
3: 5.8344631, 6.8190289, 5.8405027, 6.8146772, -0.5411301, 0.5387900
4: -11.6882181, -10.4272251, -11.6826696, -10.4358749, -0.5855820, 0.5911653
5: -2.0694301, -1.1082271, -2.0672660, -1.1128824, -0.5137264, 0.5163882
6: -9.6131115, -8.3892593, -9.6062498, -8.3861942, -0.6684903, 0.6619446
7: -7.1248455, -6.0757513, -7.1140938, -6.0857306, -0.6374021, 0.6318810
8: -2.1400084, -1.2028961, -2.1370215, -1.1950040, -0.5321238, 0.5253500
9: -4.3179069, -3.3457110, -4.3136835, -3.3541389, -0.4555459, 0.4598547

Time for backsubstitution: 22.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741259, upper bound: 0.2753195
time: 3.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741259, upper bound: 0.2753185
time: 4.10 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -8.9470282, -7.8256040, -8.9458027, -7.8257713, -0.5979295, 0.5967827
1: -7.1805220, -6.0964093, -7.1822739, -6.0968404, -0.5835426, 0.5857508
2: -2.9269843, -1.8772016, -2.9281425, -1.8770347, -0.5130661, 0.5141475
3: 5.8361044, 6.8169751, 5.8361797, 6.8167362, -0.5351751, 0.5347412
4: -11.6816206, -10.4344215, -11.6827860, -10.4342442, -0.5777431, 0.5791504
5: -2.0686359, -1.1105651, -2.0688059, -1.1114101, -0.5103019, 0.5113120
6: -9.6074333, -8.3910198, -9.6045399, -8.3908510, -0.6594293, 0.6559081
7: -7.1224642, -6.0766373, -7.1217475, -6.0766058, -0.6330452, 0.6323655
8: -2.1304073, -1.2079058, -2.1263189, -1.2079430, -0.5189567, 0.5150725
9: -4.3165040, -3.3486779, -4.3167515, -3.3500304, -0.4501787, 0.4520010

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751153
time: 5.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751164
time: 3.86 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -8.9492874, -7.8254366, -8.9507561, -7.8202028, -0.6059294, 0.6000054
1: -7.1809506, -6.0951161, -7.1857862, -6.0943575, -0.5855677, 0.5899658
2: -2.9271715, -1.8783619, -2.9271069, -1.8786478, -0.5150232, 0.5123147
3: 5.8359194, 6.8176632, 5.8356295, 6.8182287, -0.5369737, 0.5360307
4: -11.6829176, -10.4342985, -11.6853704, -10.4313431, -0.5825572, 0.5825262
5: -2.0687387, -1.1095053, -2.0719190, -1.1090616, -0.5118365, 0.5156062
6: -9.6106653, -8.3909702, -9.6122942, -8.3828154, -0.6702673, 0.6601853
7: -7.1233692, -6.0765438, -7.1242027, -6.0752268, -0.6368122, 0.6345215
8: -2.1359067, -1.2077422, -2.1377201, -1.1930895, -0.5335643, 0.5199105
9: -4.3166437, -3.3471410, -4.3204355, -3.3464541, -0.4528047, 0.4573178

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741272, upper bound: 0.2751187
time: 4.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2751195
time: 3.64 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -8.9508038, -7.8247137, -8.9470015, -7.8257575, -0.6022758, 0.5997796
1: -7.1850553, -6.0869074, -7.1844463, -6.0961699, -0.5865858, 0.5962291
2: -2.9309688, -1.8748960, -2.9295182, -1.8768148, -0.5174537, 0.5181369
3: 5.8346119, 6.8183413, 5.8360624, 6.8172388, -0.5371995, 0.5361046
4: -11.6869202, -10.4273472, -11.6850901, -10.4339666, -0.5816896, 0.5889156
5: -2.0693882, -1.1092863, -2.0690889, -1.1111367, -0.5124245, 0.5131414
6: -9.6098785, -8.3892813, -9.6050816, -8.3905926, -0.6631131, 0.6617553
7: -7.1239505, -6.0758457, -7.1219845, -6.0764780, -0.6354589, 0.6338151
8: -2.1345534, -1.2030587, -2.1280956, -1.2078094, -0.5216441, 0.5218736
9: -4.3177667, -3.3472235, -4.3171492, -3.3497527, -0.4516048, 0.4538134

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753194
time: 3.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753185
time: 3.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -8.9530592, -7.8245463, -8.9519510, -7.8201885, -0.6102765, 0.6030014
1: -7.1854868, -6.0856142, -7.1879621, -6.0936894, -0.5886099, 0.5999911
2: -2.9311554, -1.8760571, -2.9284825, -1.8784275, -0.5194108, 0.5163040
3: 5.8344274, 6.8190289, 5.8355131, 6.8187323, -0.5389984, 0.5373940
4: -11.6882181, -10.4272251, -11.6876764, -10.4310656, -0.5865042, 0.5922868
5: -2.0694921, -1.1082273, -2.0722032, -1.1087888, -0.5139587, 0.5173722
6: -9.6131144, -8.3892307, -9.6128378, -8.3825550, -0.6733941, 0.6660318
7: -7.1248550, -6.0757513, -7.1244407, -6.0750980, -0.6386435, 0.6359704
8: -2.1400528, -1.2028961, -2.1394954, -1.1929560, -0.5357947, 0.5267115
9: -4.3179069, -3.3456862, -4.3208327, -3.3461757, -0.4542305, 0.4591300

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 933
type: A, layer: 1, pos: 5805
type: A, layer: 1, pos: 5842

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 928

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741272, upper bound: 0.2753224
time: 3.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2753216
time: 4.11 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 29.37 seconds
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753164
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2739545, upper bound: 0.2753164
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2741259, upper bound: 0.2753195
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2741259, upper bound: 0.2753185
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751153
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2751164
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2741272, upper bound: 0.2751187
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2751195
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753194
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2739559, upper bound: 0.2753185
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2741272, upper bound: 0.2753224
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 29.37
Output dim: 3, lower bound: -0.2741273, upper bound: 0.2753216

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.9483805, -7.8249459, -8.9366350, -7.8352470, -0.6007693, 0.5985782
1: -7.1845870, -6.0881906, -7.1674533, -6.1115355, -0.5838361, 0.5936832
2: -2.9307642, -1.8750429, -2.9234977, -1.8809774, -0.5135341, 0.5124575
3: 5.8348465, 6.8176050, 5.8410540, 6.8131843, -0.5383567, 0.5364881
4: -11.6857195, -10.4274797, -11.6800861, -10.4387760, -0.5794070, 0.5874584
5: -2.0692158, -1.1104202, -2.0641518, -1.1152296, -0.5120317, 0.5119255
6: -9.6064091, -8.3893661, -9.5984945, -8.3942289, -0.6583033, 0.6575899
7: -7.1229692, -6.0759449, -7.1116385, -6.0871119, -0.6340301, 0.6287085
8: -2.1286130, -1.2032347, -2.1256227, -1.2098589, -0.5146503, 0.5202886
9: -4.3176064, -3.3488936, -4.3099985, -3.3577156, -0.4528143, 0.4542971

Time for backsubstitution: 21.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 5805

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737992, upper bound: 0.2753164
time: 4.09 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737992, upper bound: 0.2753164
time: 3.66 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.9533186, -7.8193703, -8.9366350, -7.8352470, -0.6057973, 0.5998614
1: -7.1881018, -6.0858994, -7.1674533, -6.1115355, -0.5859592, 0.5945427
2: -2.9297280, -1.8766530, -2.9234977, -1.8809774, -0.5128710, 0.5108845
3: 5.8343830, 6.8190975, 5.8410540, 6.8131843, -0.5390866, 0.5378841
4: -11.6880999, -10.4245825, -11.6800861, -10.4387760, -0.5819459, 0.5887959
5: -2.0723412, -1.1080737, -2.0641518, -1.1152296, -0.5128136, 0.5144083
6: -9.6141720, -8.3813267, -9.5984945, -8.3942289, -0.6657214, 0.6625347
7: -7.1254253, -6.0745320, -7.1116385, -6.0871119, -0.6356711, 0.6290778
8: -2.1400127, -1.1883612, -2.1256227, -1.2098589, -0.5233645, 0.5236475
9: -4.3212976, -3.3453162, -4.3099985, -3.3577156, -0.4565161, 0.4576584

Time for backsubstitution: 21.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5842
type: B, layer: 1, pos: 5805

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737992, upper bound: 0.2753155
time: 3.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737992, upper bound: 0.2753164
time: 3.90 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.9483805, -7.8249459, -8.9415741, -7.8296728, -0.6064353, 0.6009961
1: -7.1845870, -6.0881906, -7.1709666, -6.1092443, -0.5846951, 0.5958042
2: -2.9307642, -1.8750429, -2.9224627, -1.8825884, -0.5119598, 0.5117947
3: 5.8348465, 6.8176050, 5.8405900, 6.8146772, -0.5391743, 0.5369542
4: -11.6857195, -10.4274797, -11.6824636, -10.4358749, -0.5824454, 0.5894186
5: -2.0692158, -1.1104202, -2.0672770, -1.1128848, -0.5132204, 0.5141274
6: -9.6064091, -8.3893661, -9.6062508, -8.3861904, -0.6613833, 0.6661186
7: -7.1229692, -6.0759449, -7.1140957, -6.0856972, -0.6343994, 0.6303457
8: -2.1286130, -1.2032347, -2.1370215, -1.1949868, -0.5205619, 0.5264500
9: -4.3176064, -3.3488936, -4.3136911, -3.3541398, -0.4564610, 0.4564105

Time for backsubstitution: 21.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737475, upper bound: 0.2753194
time: 3.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737475, upper bound: 0.2753160
time: 3.51 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.9533310, -7.8193712, -8.9415846, -7.8296728, -0.6057677, 0.6045542
1: -7.1881018, -6.0857086, -7.1709666, -6.1090546, -0.5890032, 0.5973494
2: -2.9297280, -1.8766510, -2.9224627, -1.8825872, -0.5162580, 0.5151826
3: 5.8342962, 6.8190975, 5.8405027, 6.8146772, -0.5416861, 0.5395539
4: -11.6883049, -10.4245806, -11.6826706, -10.4358749, -0.5833411, 0.5913815
5: -2.0723412, -1.1080728, -2.0672781, -1.1128824, -0.5153853, 0.5145353
6: -9.6141720, -8.3813257, -9.6062498, -8.3861895, -0.6627245, 0.6620069
7: -7.1254253, -6.0745282, -7.1140938, -6.0856919, -0.6383157, 0.6326063
8: -2.1400127, -1.1883559, -2.1370215, -1.1949825, -0.5212375, 0.5268760
9: -4.3213053, -3.3453174, -4.3136964, -3.3541389, -0.4569001, 0.4583833

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 5842

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737475, upper bound: 0.2755780
time: 4.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2737475, upper bound: 0.2753157
time: 4.02 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.9446049, -7.8257856, -8.9458027, -7.8257713, -0.5954072, 0.5966368
1: -7.1800528, -6.0975957, -7.1822739, -6.0968404, -0.5829275, 0.5845613
2: -2.9267797, -1.8772650, -2.9281425, -1.8770347, -0.5126147, 0.5140760
3: 5.8363008, 6.8162384, 5.8361797, 6.8167362, -0.5342124, 0.5337284
4: -11.6804209, -10.4345570, -11.6827860, -10.4342442, -0.5763805, 0.5788124
5: -2.0685241, -1.1116996, -2.0688059, -1.1114101, -0.5101839, 0.5101728
6: -9.6039648, -8.3910751, -9.6045399, -8.3908510, -0.6557233, 0.6558292
7: -7.1214933, -6.0767398, -7.1217475, -6.0766058, -0.6320343, 0.6323287
8: -2.1245112, -1.2080803, -2.1263189, -1.2079430, -0.5130267, 0.5148476
9: -4.3163443, -3.3503249, -4.3167515, -3.3500304, -0.4500728, 0.4502667

Time for backsubstitution: 21.25 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.37 + 545.19 = 601.56 seconds
