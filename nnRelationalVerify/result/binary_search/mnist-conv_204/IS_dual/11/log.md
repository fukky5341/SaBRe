## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.41320994132
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.6942806, -5.0616770, -9.6942806, -5.0616770, -4.3366542, 4.3366537)
1: (-15.0952425, -10.8431473, -15.0952425, -10.8431473, -4.2520952, 4.2520952)
2: (-9.0615978, -5.7651587, -9.0615978, -5.7651587, -3.2964392, 3.2964392)
3: (-11.5230656, -7.4036608, -11.5230656, -7.4036608, -4.1194048, 4.1194048)
4: (-5.4777827, -1.9554484, -5.4777827, -1.9554484, -3.5223343, 3.5223343)
5: (-3.5736499, -0.4953117, -3.5736499, -0.4953117, -3.0783381, 3.0783381)
6: (-11.5837259, -6.9704914, -11.5837259, -6.9704914, -4.5754027, 4.5754023)
7: (-2.8098021, 0.8292036, -2.8098021, 0.8292036, -3.6390057, 3.6390057)
8: (-5.0775828, -1.4732656, -5.0775828, -1.4732656, -3.6043172, 3.6043172)
9: (0.4356761, 3.0562549, 0.4356761, 3.0562549, -2.6205788, 2.6205788)

## BASE Result
execution time: IAR + LP analysis = 15.33 + 33.77 = 49.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -2.1425934, upper bound: 2.1425903


# Binary Search by BASE starts (time budget: 3550.90 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.567917585372925
rel_dist={9: [-1.824091324888559, 1.824091281312575]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.44260835647583
rel_dist={9: [-1.592849651897651, 1.5928489343896768]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.3590686321258545
rel_dist={9: [-1.414766908177059, 1.4147664541724545]}

## Binary Search Result
Binary search time: 154.16 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 3396.74 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6165

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204509, upper bound: 1.8111950
time: 4.79 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240837, upper bound: 1.8240835
time: 5.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.32 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 10.32
Output dim: 9, lower bound: -1.8204509, upper bound: 1.8111950
IS_B2, status: Status.UNKNOWN, split count: 1, time: 10.32
Output dim: 9, lower bound: -1.8240837, upper bound: 1.8240835

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -9.6920967, -5.0727520, -9.6682663, -5.1634078, -3.5472002, 3.5966854
1: -15.0910149, -10.8462181, -15.0569286, -10.8714409, -3.7299900, 3.7135241
2: -9.0560865, -5.7682147, -9.0120049, -5.7965984, -2.9944596, 2.9806409
3: -11.5195675, -7.4070988, -11.4912167, -7.4395132, -4.0037174, 4.0091219
4: -5.4736834, -1.9590906, -5.4366789, -1.9887697, -3.4099574, 3.3999634
5: -3.5707574, -0.4964113, -3.5435324, -0.5056162, -2.9148788, 2.8960755
6: -11.5807791, -6.9758654, -11.5525494, -7.0190983, -3.8698578, 3.8788915
7: -2.8060970, 0.8263845, -2.7717586, 0.8029237, -3.5555577, 3.5354342
8: -5.0706773, -1.4753327, -5.0141640, -1.4970679, -3.1586003, 3.1256013
9: 0.4473801, 3.0551050, 0.5430708, 3.0429435, -2.5394819, 2.4525039

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 485

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092980, upper bound: 1.8111838
time: 4.50 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204394, upper bound: 1.8111834
time: 4.93 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -9.6942806, -5.0616770, -9.6942797, -5.0616813, -3.6005116, 3.6520977
1: -15.0952425, -10.8431473, -15.0952435, -10.8431511, -3.7727804, 3.7707851
2: -9.0615978, -5.7651587, -9.0615940, -5.7651606, -3.0377321, 3.0216036
3: -11.5230656, -7.4036608, -11.5230656, -7.4036622, -4.0477705, 4.0652876
4: -5.4777827, -1.9554484, -5.4777818, -1.9554505, -3.4424810, 3.4462752
5: -3.5736499, -0.4953117, -3.5736494, -0.4953098, -2.9307966, 2.9285197
6: -11.5837259, -6.9704914, -11.5837269, -6.9704943, -3.8962908, 3.9205384
7: -2.8098021, 0.8292036, -2.8098001, 0.8292017, -3.5922117, 3.5874515
8: -5.0775828, -1.4732656, -5.0775805, -1.4732656, -3.1915102, 3.1604235
9: 0.4356761, 3.0562549, 0.4356799, 3.0562544, -2.5760036, 2.5679135

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8222069, upper bound: 1.8240611
time: 4.85 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240613, upper bound: 1.8240610
time: 4.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.43 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.8092980, upper bound: 1.8111838
IS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.8204394, upper bound: 1.8111834
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.8222069, upper bound: 1.8240611
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 24.43
Output dim: 9, lower bound: -1.8240613, upper bound: 1.8240610

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -9.6911469, -5.0793538, -9.6469059, -5.2505517, -3.4696512, 3.5247755
1: -15.0892773, -10.8504124, -15.0200233, -10.9904060, -3.6062803, 3.6319041
2: -9.0552187, -5.7708015, -8.9961071, -5.8690658, -2.9161654, 2.9609737
3: -11.5184803, -7.4104571, -11.4747715, -7.4775038, -3.9688044, 3.9877610
4: -5.4714298, -1.9622817, -5.3929977, -2.0634356, -3.3287945, 3.3433323
5: -3.5660801, -0.4987879, -3.4133389, -0.5217376, -2.8176351, 2.7591748
6: -11.5774517, -6.9781313, -11.4602585, -7.0380478, -3.8187675, 3.7786727
7: -2.8042984, 0.8250146, -2.7443681, 0.7853341, -3.5288382, 3.5058103
8: -5.0687351, -1.4784460, -4.9956708, -1.5813994, -3.0673823, 3.0472872
9: 0.4489655, 3.0527804, 0.5750885, 2.9773242, -2.4715276, 2.3935995

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8075875, upper bound: 1.8111628
time: 4.35 seconds

## Relational analysis of IS_B1_B1_B2

### Relational analysis result of IS_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092776, upper bound: 1.8111628
time: 4.41 seconds

## BFS IS instance: IS_B1_B2

### Backsubstitution after applying IS history:
0: -9.6920967, -5.0727520, -9.6682663, -5.1634092, -3.5181804, 3.5960088
1: -15.0910149, -10.8462181, -15.0569248, -10.8714399, -3.7057714, 3.7135231
2: -9.0560865, -5.7682147, -9.0120029, -5.7965989, -2.9762511, 2.9806404
3: -11.5195675, -7.4070988, -11.4912167, -7.4395137, -4.0163569, 4.0091214
4: -5.4736834, -1.9590906, -5.4366794, -1.9887694, -3.3856630, 3.3999634
5: -3.5707574, -0.4964113, -3.5435328, -0.5056162, -2.9148779, 2.8753612
6: -11.5807791, -6.9758654, -11.5525475, -7.0190978, -3.8880105, 3.8788900
7: -2.8060970, 0.8263845, -2.7717595, 0.8029227, -3.5733423, 3.5354338
8: -5.0706773, -1.4753327, -5.0141649, -1.4970694, -3.1586008, 3.1495295
9: 0.4473801, 3.0551050, 0.5430713, 3.0429435, -2.5238523, 2.4525042

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8186755, upper bound: 1.8111629
time: 4.64 seconds

## Relational analysis of IS_B1_B2_B2

### Relational analysis result of IS_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204184, upper bound: 1.8111629
time: 4.63 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -9.6942806, -5.0616770, -9.6923923, -5.0648413, -3.5957661, 3.6486886
1: -15.0952425, -10.8431473, -15.0922251, -10.8474188, -3.7682676, 3.7677693
2: -9.0615978, -5.7651587, -9.0594234, -5.7682381, -3.0347137, 3.0200598
3: -11.5230656, -7.4036608, -11.5214596, -7.4080429, -4.0429873, 4.0632620
4: -5.4777827, -1.9554484, -5.4654350, -1.9569894, -3.4401569, 3.4317760
5: -3.5736499, -0.4953117, -3.5687125, -0.4961605, -2.9297667, 2.9226189
6: -11.5837259, -6.9704914, -11.5807705, -6.9714108, -3.8952065, 3.9172421
7: -2.8098021, 0.8292036, -2.8032002, 0.8274660, -3.5885668, 3.5783982
8: -5.0775828, -1.4732656, -5.0741301, -1.4750619, -3.1898117, 3.1572969
9: 0.4356761, 3.0562549, 0.4387255, 3.0539315, -2.5717316, 2.5647697

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 485

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8112658, upper bound: 1.8240494
time: 4.58 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221955, upper bound: 1.8240501
time: 4.57 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -9.6942787, -5.0616827, -9.7343016, -5.0527401, -3.6252441, 3.6904702
1: -15.0952377, -10.8431606, -15.1312609, -10.8247204, -3.7913995, 3.8093174
2: -9.0615921, -5.7651634, -9.0981302, -5.7384052, -3.0695448, 3.0659728
3: -11.5230627, -7.4036703, -11.5890579, -7.3757153, -4.0698705, 4.1307850
4: -5.4777594, -1.9554527, -5.5177898, -1.8448061, -3.5202403, 3.5122638
5: -3.5736399, -0.4953117, -3.6030319, -0.4554825, -2.9547772, 2.9759531
6: -11.5837221, -6.9704943, -11.5974483, -6.9417973, -3.9242382, 3.9385309
7: -2.8097863, 0.8291988, -2.8284807, 0.8835950, -3.6618099, 3.6063066
8: -5.0775743, -1.4732704, -5.0990133, -1.4518514, -3.2122021, 3.1859910
9: 0.4356833, 3.0562494, 0.3557763, 3.0763292, -2.6260190, 2.6538115

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 485

## Relational analysis of IS_B2_B2_B1

### Relational analysis result of IS_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8131062, upper bound: 1.8240499
time: 4.76 seconds

## Relational analysis of IS_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240499, upper bound: 1.8240502
time: 5.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.26 seconds
IS_B1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8075875, upper bound: 1.8111628
IS_B1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8092776, upper bound: 1.8111628
IS_B1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8186755, upper bound: 1.8111629
IS_B1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8204184, upper bound: 1.8111629
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8112658, upper bound: 1.8240494
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8221955, upper bound: 1.8240501
IS_B2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8131062, upper bound: 1.8240499
IS_B2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 25.26
Output dim: 9, lower bound: -1.8240499, upper bound: 1.8240502

## BFS IS instance: IS_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.6911469, -5.0793538, -9.6449871, -5.2536902, -3.4649010, 3.5213256
1: -15.0892773, -10.8504124, -15.0170202, -10.9946852, -3.6017513, 3.6289806
2: -9.0552187, -5.7708015, -8.9940023, -5.8721385, -2.9131775, 2.9594657
3: -11.5184803, -7.4104571, -11.4731760, -7.4818916, -3.9639921, 3.9857473
4: -5.4714298, -1.9622817, -5.3806863, -2.0649452, -3.3263941, 3.3287916
5: -3.5660801, -0.4987879, -3.4084206, -0.5225811, -2.8166919, 2.7533138
6: -11.5774517, -6.9781313, -11.4573135, -7.0389557, -3.8176966, 3.7753510
7: -2.8042984, 0.8250146, -2.7377887, 0.7836280, -3.5252476, 3.4968338
8: -5.0687351, -1.4784460, -4.9921775, -1.5831776, -3.0657635, 3.0441206
9: 0.4489655, 3.0527804, 0.5780840, 2.9749885, -2.4672432, 2.3904176

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6238

## Relational analysis of IS_B1_B1_B1_B1

### Relational analysis result of IS_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8045560, upper bound: 1.8111593
time: 4.45 seconds

## Relational analysis of IS_B1_B1_B1_B2

### Relational analysis result of IS_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8075837, upper bound: 1.8111592
time: 4.25 seconds

## BFS IS instance: IS_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.6911440, -5.0793610, -9.6868992, -5.2415962, -3.4944010, 3.5452032
1: -15.0892735, -10.8504248, -15.0554399, -10.9717865, -3.6250725, 3.6553564
2: -9.0552139, -5.7708063, -9.0321178, -5.8422904, -2.9481483, 2.9923315
3: -11.5184774, -7.4104691, -11.5407810, -7.4490948, -3.9909301, 4.0555649
4: -5.4714060, -1.9622841, -5.4324894, -1.9534919, -3.4075089, 3.3909631
5: -3.5660710, -0.4987888, -3.4421825, -0.4820280, -2.8258915, 2.8066225
6: -11.5774498, -6.9781327, -11.4740486, -7.0092382, -3.8347082, 3.7965384
7: -2.8042822, 0.8250108, -2.7624950, 0.8391423, -3.5984511, 3.5237203
8: -5.0687275, -1.4784479, -5.0169601, -1.5599804, -3.0881124, 3.0729771
9: 0.4489703, 3.0527761, 0.4957762, 2.9972131, -2.5151110, 2.4461482

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 485

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_B1_B1_B2_B1

### Relational analysis result of IS_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092716, upper bound: 1.8024260
time: 4.88 seconds

## Relational analysis of IS_B1_B1_B2_B2

### Relational analysis result of IS_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092716, upper bound: 1.8111569
time: 4.70 seconds

## BFS IS instance: IS_B1_B2_B1

### Backsubstitution after applying IS history:
0: -9.6920967, -5.0727520, -9.6663485, -5.1665683, -3.5134315, 3.5925825
1: -15.0910149, -10.8462181, -15.0539322, -10.8756943, -3.7012668, 3.7104831
2: -9.0560865, -5.7682147, -9.0098915, -5.7996840, -2.9732733, 2.9791203
3: -11.5195675, -7.4070988, -11.4896059, -7.4438982, -4.0115519, 4.0070968
4: -5.4736834, -1.9590906, -5.4243784, -1.9902941, -3.3832998, 3.3854084
5: -3.5707574, -0.4964113, -3.5385938, -0.5064659, -2.9138641, 2.8694832
6: -11.5807791, -6.9758654, -11.5496006, -7.0200157, -3.8869281, 3.8755651
7: -2.8060970, 0.8263845, -2.7651620, 0.8011975, -3.5697770, 3.5264025
8: -5.0706773, -1.4753327, -5.0106921, -1.4988356, -3.1569271, 3.1463869
9: 0.4473801, 3.0551050, 0.5460744, 3.0406232, -2.5195658, 2.4493260

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6238

## Relational analysis of IS_B1_B2_B1_B1

### Relational analysis result of IS_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8156767, upper bound: 1.8111589
time: 4.82 seconds

## Relational analysis of IS_B1_B2_B1_B2

### Relational analysis result of IS_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8186718, upper bound: 1.8111591
time: 4.87 seconds

## BFS IS instance: IS_B1_B2_B2

### Backsubstitution after applying IS history:
0: -9.6920919, -5.0727587, -9.7082787, -5.1544108, -3.5429831, 3.6164057
1: -15.0910091, -10.8462315, -15.0922909, -10.8529510, -3.7244205, 3.7514215
2: -9.0560818, -5.7682214, -9.0486269, -5.7698059, -3.0084000, 3.0250559
3: -11.5195637, -7.4071088, -11.5572824, -7.4113998, -4.0383906, 4.0760098
4: -5.4736614, -1.9590943, -5.4759712, -1.8787695, -3.4639587, 3.4659548
5: -3.5707483, -0.4964113, -3.5723066, -0.4659138, -2.9378471, 2.9230270
6: -11.5807753, -6.9758663, -11.5663719, -6.9903316, -3.9158702, 3.8967686
7: -2.8060822, 0.8263812, -2.7901073, 0.8567567, -3.6429968, 3.5541201
8: -5.0706701, -1.4753351, -5.0355597, -1.4757347, -3.1792107, 3.1752818
9: 0.4473858, 3.0551000, 0.4640450, 3.0631425, -2.5671432, 2.5359852

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6238

## Relational analysis of IS_B1_B2_B2_B1

### Relational analysis result of IS_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8174184, upper bound: 1.8111591
time: 4.70 seconds

## Relational analysis of IS_B1_B2_B2_B2

### Relational analysis result of IS_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204146, upper bound: 1.8111591
time: 4.75 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -9.6933336, -5.0682793, -9.6712322, -5.1520877, -3.5181003, 3.6057725
1: -15.0935040, -10.8473444, -15.0553341, -10.9664803, -3.6444788, 3.6910558
2: -9.0607300, -5.7677436, -9.0435333, -5.8407331, -2.9560308, 3.0003989
3: -11.5219812, -7.4070048, -11.5050220, -7.4456034, -4.0083637, 4.0418501
4: -5.4755297, -1.9586420, -5.4216862, -2.0318365, -3.3589230, 3.3807096
5: -3.5689740, -0.4976864, -3.4384532, -0.5122719, -2.8320565, 2.7856517
6: -11.5804024, -6.9727602, -11.4884567, -6.9903512, -3.8437967, 3.8172059
7: -2.8080020, 0.8278317, -2.7756500, 0.8098340, -3.5617704, 3.5484548
8: -5.0756407, -1.4763794, -5.0552340, -1.5594053, -3.0983739, 3.0788431
9: 0.4372597, 3.0539303, 0.4704814, 2.9882627, -2.5037565, 2.5061202

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6238

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8083676, upper bound: 1.8240464
time: 4.55 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8112621, upper bound: 1.8240459
time: 5.65 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -9.6942806, -5.0616770, -9.6923933, -5.0648437, -3.5667458, 3.6486881
1: -15.0952425, -10.8431473, -15.0922241, -10.8474178, -3.7440481, 3.7677698
2: -9.0615978, -5.7651587, -9.0594244, -5.7682376, -3.0165071, 3.0200596
3: -11.5230656, -7.4036608, -11.5214596, -7.4080424, -4.0556254, 4.0632629
4: -5.4777827, -1.9554484, -5.4654360, -1.9569901, -3.4158602, 3.4317765
5: -3.5736499, -0.4953117, -3.5687113, -0.4961615, -2.9297657, 2.9019043
6: -11.5837259, -6.9704914, -11.5807724, -6.9714093, -3.9133582, 3.9172411
7: -2.8098021, 0.8292036, -2.8032002, 0.8274655, -3.6063099, 3.5783968
8: -5.0775828, -1.4732656, -5.0741301, -1.4750619, -3.1898108, 3.1812248
9: 0.4356761, 3.0562549, 0.4387255, 3.0539317, -2.5561023, 2.5647702

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B2_B1_B2_B1

### Relational analysis result of IS_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8210535, upper bound: 1.8240481
time: 4.76 seconds

## Relational analysis of IS_B2_B1_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221936, upper bound: 1.8240481
time: 4.70 seconds

## BFS IS instance: IS_B2_B2_B1

### Backsubstitution after applying IS history:
0: -9.6933298, -5.0682869, -9.7131281, -5.1400528, -3.5475636, 3.6298442
1: -15.0935011, -10.8473549, -15.0944338, -10.9436245, -3.6677933, 3.7174697
2: -9.0607262, -5.7677498, -9.0815487, -5.8109303, -2.9909563, 3.0323896
3: -11.5219793, -7.4070144, -11.5725479, -7.4129944, -4.0353661, 4.1103201
4: -5.4755063, -1.9586427, -5.4742746, -1.9196887, -3.4394188, 3.4432750
5: -3.5689645, -0.4976892, -3.4728603, -0.4715900, -2.8411236, 2.8389437
6: -11.5803967, -6.9727612, -11.5050917, -6.9607058, -3.8609257, 3.8383999
7: -2.8079882, 0.8278279, -2.8006687, 0.8659363, -3.6341305, 3.5755777
8: -5.0756330, -1.4763837, -5.0800457, -1.5361023, -3.1208253, 3.1075063
9: 0.4372649, 3.0539262, 0.3871574, 3.0103724, -2.5579929, 2.5630336

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6238

## Relational analysis of IS_B2_B2_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8102064, upper bound: 1.8240468
time: 4.74 seconds

## Relational analysis of IS_B2_B2_B1_B2

### Relational analysis result of IS_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8131025, upper bound: 1.8240464
time: 4.86 seconds

## BFS IS instance: IS_B2_B2_B2

### Backsubstitution after applying IS history:
0: -9.6942787, -5.0616827, -9.7343035, -5.0527396, -3.5962248, 3.6904700
1: -15.0952377, -10.8431606, -15.1312618, -10.8247185, -3.7671814, 3.8093164
2: -9.0615921, -5.7651634, -9.0981312, -5.7384057, -3.0513382, 3.0659723
3: -11.5230627, -7.4036703, -11.5890579, -7.3757153, -4.0825100, 4.1307850
4: -5.4777594, -1.9554527, -5.5177894, -1.8448076, -3.4959364, 3.5122633
5: -3.5736399, -0.4953117, -3.6030331, -0.4554825, -2.9530993, 2.9552391
6: -11.5837221, -6.9704943, -11.5974483, -6.9417963, -3.9423923, 3.9385309
7: -2.8097863, 0.8291988, -2.8284793, 0.8835959, -3.6794858, 3.6063066
8: -5.0775743, -1.4732704, -5.0990119, -1.4518509, -3.2122021, 3.2099195
9: 0.4356833, 3.0562494, 0.3557758, 3.0763297, -2.6103902, 2.6538124

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6238

## Relational analysis of IS_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8212188, upper bound: 1.8240480
time: 6.05 seconds

## Relational analysis of IS_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240463, upper bound: 1.8240463
time: 4.71 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 25.67 seconds
IS_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8045560, upper bound: 1.8111593
IS_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8075837, upper bound: 1.8111592
IS_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8092716, upper bound: 1.8024260
IS_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8092716, upper bound: 1.8111569
IS_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8156767, upper bound: 1.8111589
IS_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8186718, upper bound: 1.8111591
IS_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8174184, upper bound: 1.8111591
IS_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8204146, upper bound: 1.8111591
IS_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8083676, upper bound: 1.8240464
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8112621, upper bound: 1.8240459
IS_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8210535, upper bound: 1.8240481
IS_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8221936, upper bound: 1.8240481
IS_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8102064, upper bound: 1.8240468
IS_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8131025, upper bound: 1.8240464
IS_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8212188, upper bound: 1.8240480
IS_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 25.67
Output dim: 9, lower bound: -1.8240463, upper bound: 1.8240463

## BFS IS instance: IS_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.6909313, -5.0794134, -9.6117172, -5.2645445, -3.4509554, 3.4904156
1: -15.0892429, -10.8505020, -15.0111618, -11.0076971, -3.5897422, 3.6225796
2: -9.0550156, -5.7708464, -8.9620314, -5.8797216, -2.9053097, 2.9286065
3: -11.5184460, -7.4105005, -11.4674473, -7.4886365, -3.9576368, 3.9779592
4: -5.4713240, -1.9623665, -5.3626528, -2.0780799, -3.3157930, 3.3066835
5: -3.5659137, -0.4988022, -3.3822341, -0.5248423, -2.8026848, 2.7285395
6: -11.5771065, -6.9781604, -11.4028711, -7.0441961, -3.7789259, 3.7235312
7: -2.8042159, 0.8249574, -2.7235060, 0.7750325, -3.5196981, 3.4784093
8: -5.0687151, -1.4785366, -4.9885406, -1.5975599, -3.0530024, 3.0397184
9: 0.4490552, 3.0526690, 0.5932865, 2.9575026, -2.4520876, 2.3680310

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_B1_B1_B1_B1_A1

### Relational analysis result of IS_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8045560, upper bound: 1.8094288
time: 4.38 seconds

## Relational analysis of IS_B1_B1_B1_B1_A2

### Relational analysis result of IS_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8045560, upper bound: 1.8111593
time: 4.44 seconds

## BFS IS instance: IS_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.6911469, -5.0793538, -9.6449862, -5.2536879, -3.4649000, 3.5079598
1: -15.0892773, -10.8504124, -15.0170183, -10.9946842, -3.6032820, 3.6289372
2: -9.0552187, -5.7708015, -8.9940023, -5.8721399, -2.9131699, 2.9471636
3: -11.5184803, -7.4104571, -11.4731750, -7.4818912, -3.9725151, 3.9857340
4: -5.4714298, -1.9622817, -5.3806839, -2.0649457, -3.3251209, 3.3287363
5: -3.5660801, -0.4987879, -3.4084210, -0.5225816, -2.8166432, 2.7468872
6: -11.5774517, -6.9781313, -11.4573116, -7.0389557, -3.8175793, 3.7429695
7: -2.8042984, 0.8250146, -2.7377882, 0.7836275, -3.5264926, 3.4968324
8: -5.0687351, -1.4784460, -4.9921789, -1.5831785, -3.0664597, 3.0441213
9: 0.4489655, 3.0527804, 0.5780849, 2.9749887, -2.4602833, 2.3903680

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6238

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_B1_B1_B1_B2_A1

### Relational analysis result of IS_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8075837, upper bound: 1.8094287
time: 4.59 seconds

## Relational analysis of IS_B1_B1_B1_B2_A2

### Relational analysis result of IS_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8075837, upper bound: 1.8111592
time: 4.38 seconds

## BFS IS instance: IS_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -9.6876373, -5.1020947, -9.6489468, -5.4238067, -3.3028421, 3.3758655
1: -15.0847034, -10.8536568, -15.0245857, -11.0003738, -3.5932422, 3.5994184
2: -9.0522175, -5.7910647, -8.9993353, -6.0047326, -2.7761979, 2.7741165
3: -11.5172081, -7.4298892, -11.5245895, -7.5992475, -3.7556839, 3.9624476
4: -5.4586916, -1.9646955, -5.3291569, -1.9771031, -3.3234925, 3.2791958
5: -3.5591607, -0.4999075, -3.3900747, -0.4938011, -2.7915468, 2.7320788
6: -11.5743589, -7.0073376, -11.4364204, -7.2433615, -3.5972972, 3.5716846
7: -2.7897472, 0.8224401, -2.6446238, 0.8126101, -3.5112667, 3.4014497
8: -5.0649581, -1.4965887, -4.9820623, -1.7055397, -2.9388399, 2.9343338
9: 0.4576645, 3.0510824, 0.5672393, 2.9814196, -2.4523149, 2.3665595

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6238

## Relational analysis of IS_B1_B1_B2_B1_A1

### Relational analysis result of IS_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092677, upper bound: 1.7994272
time: 5.22 seconds

## Relational analysis of IS_B1_B1_B2_B1_A2

### Relational analysis result of IS_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092679, upper bound: 1.8024222
time: 4.90 seconds

## BFS IS instance: IS_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -9.6911440, -5.0793610, -9.6869001, -5.2416072, -3.3727121, 3.5360394
1: -15.0892735, -10.8504248, -15.0554371, -10.9717865, -3.6250706, 3.6664083
2: -9.0552139, -5.7708063, -9.0321150, -5.8422971, -2.8661575, 2.9849029
3: -11.5184774, -7.4104691, -11.5407820, -7.4490995, -3.9622469, 4.0501809
4: -5.4714060, -1.9622841, -5.4324851, -1.9534926, -3.4218969, 3.3909605
5: -3.5660710, -0.4987888, -3.4421797, -0.4820271, -2.8240795, 2.8019502
6: -11.5774498, -6.9781327, -11.4740458, -7.0092440, -3.6968369, 3.7955742
7: -2.8042822, 0.8250108, -2.7624888, 0.8391423, -3.6040878, 3.5237155
8: -5.0687275, -1.4784479, -5.0169578, -1.5599847, -3.0600672, 3.0729754
9: 0.4489703, 3.0527761, 0.4957800, 2.9972126, -2.5122986, 2.4455640

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6238

## Relational analysis of IS_B1_B1_B2_B2_A1

### Relational analysis result of IS_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092678, upper bound: 1.8081463
time: 4.25 seconds

## Relational analysis of IS_B1_B1_B2_B2_A2

### Relational analysis result of IS_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8092678, upper bound: 1.8111529
time: 4.47 seconds

## BFS IS instance: IS_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -9.6918831, -5.0728102, -9.6330090, -5.1774197, -3.4993134, 3.5615377
1: -15.0909796, -10.8463078, -15.0480928, -10.8887320, -3.6893144, 3.7039924
2: -9.0558825, -5.7682629, -8.9778814, -5.8072562, -2.9653568, 2.9482477
3: -11.5195293, -7.4071388, -11.4838829, -7.4505835, -4.0052524, 3.9992990
4: -5.4735785, -1.9591758, -5.4065123, -2.0033937, -3.3727179, 3.3658094
5: -3.5705895, -0.4964247, -3.5123129, -0.5087223, -2.9104910, 2.8448193
6: -11.5804300, -6.9758949, -11.4952869, -7.0252423, -3.8803525, 3.8239484
7: -2.8060155, 0.8263292, -2.7508574, 0.7926054, -3.5642066, 3.5079093
8: -5.0706549, -1.4754233, -5.0071349, -1.5131564, -3.1438551, 3.1420379
9: 0.4474702, 3.0549927, 0.5611916, 3.0231578, -2.5044222, 2.4340122

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B1_B2_B1_B1_B1

### Relational analysis result of IS_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8149682, upper bound: 1.8111579
time: 4.61 seconds

## Relational analysis of IS_B1_B2_B1_B1_B2

### Relational analysis result of IS_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8156755, upper bound: 1.8111579
time: 5.17 seconds

## BFS IS instance: IS_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -9.6920967, -5.0727520, -9.6663446, -5.1665673, -3.5134325, 3.5791824
1: -15.0910149, -10.8462181, -15.0539284, -10.8756924, -3.7027974, 3.7104840
2: -9.0560865, -5.7682147, -9.0098915, -5.7996840, -2.9732656, 2.9667630
3: -11.5195675, -7.4070988, -11.4896059, -7.4438982, -4.0200748, 4.0070834
4: -5.4736834, -1.9590906, -5.4243784, -1.9902937, -3.3820524, 3.3854094
5: -3.5707574, -0.4964113, -3.5385933, -0.5064659, -2.9138641, 2.8630567
6: -11.5807791, -6.9758654, -11.5496016, -7.0200157, -3.8869281, 3.8432770
7: -2.8060970, 0.8263845, -2.7651620, 0.8011990, -3.5710206, 3.5264020
8: -5.0706773, -1.4753327, -5.0106926, -1.4988360, -3.1575956, 3.1463866
9: 0.4473801, 3.0551050, 0.5460744, 3.0406237, -2.5125952, 2.4493260

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B1_B2_B1_B2_B1

### Relational analysis result of IS_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8179672, upper bound: 1.8111579
time: 4.77 seconds

## Relational analysis of IS_B1_B2_B1_B2_B2

### Relational analysis result of IS_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8186706, upper bound: 1.8111577
time: 4.68 seconds

## BFS IS instance: IS_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -9.6918812, -5.0728202, -9.6749392, -5.1652346, -3.5288448, 3.5852883
1: -15.0909729, -10.8463173, -15.0864048, -10.8659916, -3.7124529, 3.7449360
2: -9.0558777, -5.7682672, -9.0164661, -5.7773647, -3.0004368, 2.9941256
3: -11.5195255, -7.4071498, -11.5515337, -7.4180670, -4.0320797, 4.0681915
4: -5.4735551, -1.9591773, -5.4580693, -1.8917444, -3.4535971, 3.4460244
5: -3.5705798, -0.4964266, -3.5460114, -0.4681749, -2.9238620, 2.8980944
6: -11.5804262, -6.9758987, -11.5120554, -6.9955692, -3.8972030, 3.8449941
7: -2.8060007, 0.8263259, -2.7757316, 0.8481979, -3.6374068, 3.5355988
8: -5.0706482, -1.4754272, -5.0319977, -1.4901237, -3.1660700, 3.1709011
9: 0.4474759, 3.0549884, 0.4791799, 3.0456729, -2.5521164, 2.5158629

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B1_B2_B2_B1_B1

### Relational analysis result of IS_B1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8166999, upper bound: 1.8111580
time: 4.57 seconds

## Relational analysis of IS_B1_B2_B2_B1_B2

### Relational analysis result of IS_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8174172, upper bound: 1.8111579
time: 4.69 seconds

## BFS IS instance: IS_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -9.6920919, -5.0727587, -9.7082787, -5.1544104, -3.5429831, 3.6030109
1: -15.0910091, -10.8462315, -15.0922918, -10.8529530, -3.7259507, 3.7514219
2: -9.0560818, -5.7682214, -9.0486269, -5.7698035, -3.0083923, 3.0126617
3: -11.5195637, -7.4071088, -11.5572834, -7.4114013, -4.0469131, 4.0759888
4: -5.4736614, -1.9590943, -5.4759722, -1.8787688, -3.4628258, 3.4659543
5: -3.5707483, -0.4964113, -3.5723066, -0.4659128, -2.9377975, 2.9166000
6: -11.5807753, -6.9758663, -11.5663719, -6.9903312, -3.9158697, 3.8644629
7: -2.8060822, 0.8263812, -2.7901073, 0.8567567, -3.6442399, 3.5541201
8: -5.0706701, -1.4753351, -5.0355592, -1.4757338, -3.1798382, 3.1752818
9: 0.4473858, 3.0551000, 0.4640455, 3.0631425, -2.5601768, 2.5359855

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B1_B2_B2_B2_B1

### Relational analysis result of IS_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8197121, upper bound: 1.8111579
time: 4.68 seconds

## Relational analysis of IS_B1_B2_B2_B2_B2

### Relational analysis result of IS_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8204134, upper bound: 1.8111578
time: 4.65 seconds

## BFS IS instance: IS_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -9.6931171, -5.0683398, -9.6376772, -5.1629086, -3.5040274, 3.5746045
1: -15.0934687, -10.8474293, -15.0494328, -10.9795265, -3.6325140, 3.6846330
2: -9.0605268, -5.7677889, -9.0117407, -5.8483229, -2.9480963, 2.9694026
3: -11.5219469, -7.4070463, -11.4993258, -7.4524379, -4.0018964, 4.0340052
4: -5.4754248, -1.9587253, -5.4035339, -2.0449104, -3.3483801, 3.3587508
5: -3.5688071, -0.4977007, -3.4121442, -0.5145340, -2.8180580, 2.7607720
6: -11.5800552, -6.9727898, -11.4337940, -6.9955950, -3.8049793, 3.7651386
7: -2.8079224, 0.8277760, -2.7612505, 0.8012896, -3.5561686, 3.5298657
8: -5.0756183, -1.4764719, -5.0515680, -1.5740695, -3.0853114, 3.0744228
9: 0.4373507, 3.0538177, 0.4856462, 2.9707034, -2.4884949, 2.4837060

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B2_B1_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8058767, upper bound: 1.8240336
time: 4.52 seconds

## Relational analysis of IS_B2_B1_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8083530, upper bound: 1.8240334
time: 4.90 seconds

## BFS IS instance: IS_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -9.6933336, -5.0682793, -9.6712341, -5.1520891, -3.5181012, 3.5924673
1: -15.0935040, -10.8473444, -15.0553322, -10.9664822, -3.6460099, 3.6910131
2: -9.0607300, -5.7677436, -9.0435324, -5.8407340, -2.9560232, 2.9881537
3: -11.5219812, -7.4070048, -11.5050220, -7.4456000, -4.0168848, 4.0418367
4: -5.4755297, -1.9586420, -5.4216857, -2.0318367, -3.3577337, 3.3806543
5: -3.5689740, -0.4976864, -3.4384532, -0.5122719, -2.8320084, 2.7792246
6: -11.5804024, -6.9727602, -11.4884548, -6.9903517, -3.8436785, 3.7848506
7: -2.8080020, 0.8278317, -2.7756505, 0.8098345, -3.5630136, 3.5484543
8: -5.0756407, -1.4763794, -5.0552349, -1.5594063, -3.0991158, 3.0788431
9: 0.4372597, 3.0539303, 0.4704819, 2.9882631, -2.4968321, 2.5060711

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6238

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B2_B1_B1_B2_B1

### Relational analysis result of IS_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8100802, upper bound: 1.8240449
time: 5.16 seconds

## Relational analysis of IS_B2_B1_B1_B2_B2

### Relational analysis result of IS_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8112601, upper bound: 1.8240444
time: 5.21 seconds

## BFS IS instance: IS_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -9.6930370, -5.0620308, -9.6512356, -5.0780897, -3.5543084, 3.6180382
1: -15.0937977, -10.8437061, -15.0474968, -10.8663874, -3.7178001, 3.7127666
2: -9.0610828, -5.7666473, -9.0413675, -5.8187485, -2.9710798, 2.9981716
3: -11.5224361, -7.4047508, -11.5053825, -7.4457655, -4.0045819, 4.0452304
4: -5.4765244, -1.9561632, -5.4229603, -1.9768759, -3.3977394, 3.3895550
5: -3.5719020, -0.4957018, -3.5105958, -0.5057306, -2.8908081, 2.8385034
6: -11.5817223, -6.9709616, -11.5134363, -6.9838171, -3.8713260, 3.8453398
7: -2.8086576, 0.8286805, -2.7644582, 0.8162532, -3.5708671, 3.5386677
8: -5.0766315, -1.4743919, -5.0523324, -1.5129466, -3.1490374, 3.0769360
9: 0.4363089, 3.0557566, 0.4601135, 3.0372555, -2.5375686, 2.5251789

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6238

## Relational analysis of IS_B2_B1_B2_B1_B1

### Relational analysis result of IS_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8181897, upper bound: 1.8240449
time: 4.78 seconds

## Relational analysis of IS_B2_B1_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8210498, upper bound: 1.8240445
time: 4.66 seconds

## BFS IS instance: IS_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -9.6942806, -5.0616770, -9.6923904, -5.0648417, -3.5667453, 3.6451993
1: -15.0952425, -10.8431473, -15.0922222, -10.8474178, -3.7440476, 3.7694740
2: -9.0615978, -5.7651587, -9.0594234, -5.7682390, -3.0148764, 3.0200601
3: -11.5230656, -7.4036608, -11.5214605, -7.4080462, -4.0554204, 4.0844812
4: -5.4777827, -1.9554484, -5.4654350, -1.9569906, -3.4157944, 3.4307365
5: -3.5736499, -0.4953117, -3.5687113, -0.4961615, -2.9586415, 2.9019036
6: -11.5837259, -6.9704914, -11.5807724, -6.9714098, -3.9133606, 3.9057431
7: -2.8098021, 0.8292036, -2.8031993, 0.8274646, -3.6150732, 3.5783978
8: -5.0775828, -1.4732656, -5.0741305, -1.4750624, -3.1898098, 3.2328298
9: 0.4356761, 3.0562549, 0.4387264, 3.0539308, -2.5550361, 2.5647695

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B2_B1_B2_B2_B1

### Relational analysis result of IS_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8194933, upper bound: 1.8240371
time: 5.33 seconds

## Relational analysis of IS_B2_B1_B2_B2_B2

### Relational analysis result of IS_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8221796, upper bound: 1.8240353
time: 4.69 seconds

## BFS IS instance: IS_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -9.6931181, -5.0683470, -9.6795712, -5.1508498, -3.5334706, 3.5987668
1: -15.0934610, -10.8474426, -15.0884876, -10.9566679, -3.6558228, 3.7110672
2: -9.0605221, -5.7677956, -9.0495758, -5.8185062, -2.9829969, 3.0017035
3: -11.5219440, -7.4070563, -11.5668221, -7.4198074, -4.0288868, 4.1024652
4: -5.4754004, -1.9587284, -5.4560690, -1.9326823, -3.4292164, 3.4212162
5: -3.5687973, -0.4977036, -3.4465353, -0.4738588, -2.8271308, 2.8139288
6: -11.5800505, -6.9727926, -11.4504290, -6.9659543, -3.8221650, 3.7863102
7: -2.8079071, 0.8277721, -2.7862639, 0.8574238, -3.6284666, 3.5569453
8: -5.0756116, -1.4764743, -5.0763707, -1.5508318, -3.1076951, 3.1030564
9: 0.4373565, 3.0538135, 0.4023352, 2.9928079, -2.5427237, 2.5405779

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_B2_B2_B1_B1_B1

### Relational analysis result of IS_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8102003, upper bound: 1.8152244
time: 4.64 seconds

## Relational analysis of IS_B2_B2_B1_B1_B2

### Relational analysis result of IS_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8102004, upper bound: 1.8240405
time: 4.97 seconds

## BFS IS instance: IS_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -9.6933298, -5.0682869, -9.7131290, -5.1400528, -3.5475636, 3.6165442
1: -15.0935011, -10.8473549, -15.0944338, -10.9436255, -3.6693249, 3.7174273
2: -9.0607262, -5.7677498, -9.0815477, -5.8109303, -2.9909506, 3.0200679
3: -11.5219793, -7.4070144, -11.5725479, -7.4129939, -4.0438862, 4.1103001
4: -5.4755063, -1.9586427, -5.4742761, -1.9196885, -3.4383392, 3.4432206
5: -3.5689645, -0.4976892, -3.4728613, -0.4715900, -2.8410749, 2.8325176
6: -11.5803967, -6.9727612, -11.5050926, -6.9607038, -3.8608074, 3.8060298
7: -2.8079882, 0.8278279, -2.8006692, 0.8659339, -3.6353865, 3.5755782
8: -5.0756330, -1.4763837, -5.0800457, -1.5361023, -3.1215248, 3.1075063
9: 0.4372649, 3.0539262, 0.3871570, 3.0103726, -2.5510664, 2.5629847

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6238

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6208

## Relational analysis of IS_B2_B2_B1_B2_B1

### Relational analysis result of IS_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8130964, upper bound: 1.8152246
time: 4.51 seconds

## Relational analysis of IS_B2_B2_B1_B2_B2

### Relational analysis result of IS_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8130964, upper bound: 1.8240401
time: 4.62 seconds

## BFS IS instance: IS_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -9.6940651, -5.0617437, -9.7006807, -5.0635319, -3.5819721, 3.6591146
1: -15.0952015, -10.8432446, -15.1253366, -10.8377876, -3.7552967, 3.8028250
2: -9.0613890, -5.7652092, -9.0661221, -5.7459731, -3.0433307, 3.0349016
3: -11.5230293, -7.4037132, -11.5833397, -7.3824692, -4.0760889, 4.1229191
4: -5.4776540, -1.9555365, -5.4997673, -1.8577192, -3.4857483, 3.4924324
5: -3.5734730, -0.4953241, -3.5766907, -0.4577484, -2.9391232, 2.9302094
6: -11.5833750, -6.9705286, -11.5429087, -6.9470358, -3.9232941, 3.8865104
7: -2.8097062, 0.8291426, -2.8140593, 0.8750882, -3.6738505, 3.5876188
8: -5.0775528, -1.4733620, -5.0954213, -1.4665179, -3.1987748, 3.2055254
9: 0.4357715, 3.0561376, 0.3708658, 3.0587854, -2.5951345, 2.6324127

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B2_B2_B2_B1_B1

### Relational analysis result of IS_B2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8200497, upper bound: 1.8240447
time: 4.52 seconds

## Relational analysis of IS_B2_B2_B2_B1_B2

### Relational analysis result of IS_B2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8212169, upper bound: 1.8111577
time: 10.42 seconds

## BFS IS instance: IS_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -9.6942787, -5.0616827, -9.7342997, -5.0527401, -3.5962248, 3.6771927
1: -15.0952377, -10.8431606, -15.1312618, -10.8247194, -3.7687120, 3.8093171
2: -9.0615921, -5.7651634, -9.0981312, -5.7384057, -3.0513315, 3.0536370
3: -11.5230627, -7.4036703, -11.5890589, -7.3757157, -4.0910287, 4.1307640
4: -5.4777594, -1.9554527, -5.5177894, -1.8448071, -3.4948759, 3.5122640
5: -3.5736399, -0.4953117, -3.6030312, -0.4554825, -2.9530501, 2.9488127
6: -11.5837221, -6.9704943, -11.5974474, -6.9417963, -3.9423904, 3.9062500
7: -2.8097863, 0.8291988, -2.8284807, 0.8835955, -3.6807299, 3.6063056
8: -5.0775743, -1.4732704, -5.0990133, -1.4518533, -3.2128739, 3.2099199
9: 0.4356833, 3.0562494, 0.3557758, 3.0763302, -2.6034536, 2.6538117

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6165
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_B2_B2_B2_B2_B1

### Relational analysis result of IS_B2_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8229118, upper bound: 1.8240445
time: 4.83 seconds

## Relational analysis of IS_B2_B2_B2_B2_B2

### Relational analysis result of IS_B2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.8240445, upper bound: 1.8240447
time: 4.45 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.10 seconds
IS_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8045560, upper bound: 1.8094288
IS_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8045560, upper bound: 1.8111593
IS_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8075837, upper bound: 1.8094287
IS_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8075837, upper bound: 1.8111592
IS_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8092677, upper bound: 1.7994272
IS_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8092679, upper bound: 1.8024222
IS_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8092678, upper bound: 1.8081463
IS_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8092678, upper bound: 1.8111529
IS_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8149682, upper bound: 1.8111579
IS_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8156755, upper bound: 1.8111579
IS_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8179672, upper bound: 1.8111579
IS_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8186706, upper bound: 1.8111577
IS_B1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8166999, upper bound: 1.8111580
IS_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8174172, upper bound: 1.8111579
IS_B1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8197121, upper bound: 1.8111579
IS_B1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8204134, upper bound: 1.8111578
IS_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8058767, upper bound: 1.8240336
IS_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8083530, upper bound: 1.8240334
IS_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8100802, upper bound: 1.8240449
IS_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8112601, upper bound: 1.8240444
IS_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8181897, upper bound: 1.8240449
IS_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8210498, upper bound: 1.8240445
IS_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8194933, upper bound: 1.8240371
IS_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8221796, upper bound: 1.8240353
IS_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8102003, upper bound: 1.8152244
IS_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8102004, upper bound: 1.8240405
IS_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8130964, upper bound: 1.8152246
IS_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8130964, upper bound: 1.8240401
IS_B2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8200497, upper bound: 1.8240447
IS_B2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8212169, upper bound: 1.8111577
IS_B2_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8229118, upper bound: 1.8240445
IS_B2_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.10
Output dim: 9, lower bound: -1.8240445, upper bound: 1.8240447

## BFS IS instance: IS_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6890392, -5.0825768, -9.6117172, -5.2645445, -3.4475470, 3.4856796
1: -15.0862255, -10.8547688, -15.0111618, -11.0076971, -3.5867267, 3.6180482
2: -9.0528984, -5.7739277, -8.9620314, -5.8797216, -2.9038010, 2.9255877
3: -11.5168381, -7.4148622, -11.4674473, -7.4886365, -3.9556398, 3.9731698
4: -5.4589829, -1.9639035, -5.3626528, -2.0780799, -3.3012857, 3.3043556
5: -3.5609760, -0.4996519, -3.3822341, -0.5248423, -2.7967916, 2.7275097
6: -11.5741539, -6.9790764, -11.4028711, -7.0441961, -3.7755852, 3.7224498
7: -2.7976193, 0.8232226, -2.7235060, 0.7750325, -3.5107079, 3.4748197
8: -5.0652595, -1.4803309, -4.9885406, -1.5975599, -3.0498800, 3.0380416
9: 0.4520960, 3.0503464, 0.5932865, 2.9575026, -2.4489510, 2.3637466

Time for backsubstitution: 14.64 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.567917585372925
rel_dist={9: [-1.824091324888559, 1.824091281312575]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863984, upper bound: 1.5913667
time: 4.58 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928436, upper bound: 1.5928428
time: 4.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.77 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.77
Output dim: 9, lower bound: -1.5863984, upper bound: 1.5913667
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.77
Output dim: 9, lower bound: -1.5928436, upper bound: 1.5928428

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -9.6682663, -5.1634078, -9.6897707, -5.0843816, -3.2358227, 3.2024789
1: -15.0569286, -10.8714409, -15.0866251, -10.8494310, -3.3780842, 3.3913188
2: -9.0120049, -5.7965984, -9.0504446, -5.7714515, -2.7872477, 2.7986658
3: -11.4912167, -7.4395132, -11.5159016, -7.4107132, -3.7235203, 3.7175555
4: -5.4366789, -1.9887697, -5.4693594, -1.9629092, -3.2290359, 3.2397294
5: -3.5435324, -0.5056162, -3.5676818, -0.4975634, -2.6709723, 2.6876085
6: -11.5525494, -7.0190983, -11.5776548, -6.9814577, -3.5449133, 3.5388389
7: -2.7717586, 0.8029237, -2.8021483, 0.8234305, -3.3642645, 3.3844867
8: -5.0141640, -1.4970679, -5.0634212, -1.4775352, -2.8480310, 2.8756852
9: 0.5430708, 3.0429435, 0.4596386, 3.0538754, -2.3258138, 2.4002707

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863801, upper bound: 1.5897746
time: 4.78 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863800, upper bound: 1.5913484
time: 4.68 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.6942797, -5.0616813, -9.6942806, -5.0616770, -3.3098207, 3.2496691
1: -15.0952435, -10.8431511, -15.0952425, -10.8431473, -3.4391594, 3.4401548
2: -9.0615940, -5.7651606, -9.0615978, -5.7651587, -2.8275814, 2.8482616
3: -11.5230656, -7.4036622, -11.5230656, -7.4036608, -3.7780285, 3.7658453
4: -5.4777818, -1.9554505, -5.4777827, -1.9554484, -3.2804255, 3.2734942
5: -3.5736494, -0.4953098, -3.5736499, -0.4953117, -2.7044544, 2.7053530
6: -11.5837269, -6.9704943, -11.5837259, -6.9704914, -3.5931077, 3.5633874
7: -2.8098001, 0.8292017, -2.8098021, 0.8292036, -3.4212871, 3.4224801
8: -5.0775805, -1.4732656, -5.0775828, -1.4732656, -2.8788519, 2.9163718
9: 0.4356799, 3.0562544, 0.4356761, 3.0562549, -2.4426041, 2.4498577

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928246, upper bound: 1.5911337
time: 4.71 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928246, upper bound: 1.5928234
time: 5.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.54 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 24.54
Output dim: 9, lower bound: -1.5863801, upper bound: 1.5897746
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 24.54
Output dim: 9, lower bound: -1.5863800, upper bound: 1.5913484
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 24.54
Output dim: 9, lower bound: -1.5928246, upper bound: 1.5911337
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 24.54
Output dim: 9, lower bound: -1.5928246, upper bound: 1.5928234

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -9.6663475, -5.1665659, -9.6897707, -5.0843816, -3.2323961, 3.1977305
1: -15.0539322, -10.8756943, -15.0866251, -10.8494310, -3.3750439, 3.3868132
2: -9.0098915, -5.7996831, -9.0504446, -5.7714515, -2.7857275, 2.7956860
3: -11.4896049, -7.4438972, -11.5159016, -7.4107132, -3.7214937, 3.7127509
4: -5.4243779, -1.9902939, -5.4693594, -1.9629092, -3.2144809, 3.2373662
5: -3.5385942, -0.5064659, -3.5676818, -0.4975634, -2.6650939, 2.6865933
6: -11.5496006, -7.0200167, -11.5776548, -6.9814577, -3.5415869, 3.5377569
7: -2.7651610, 0.8011990, -2.8021483, 0.8234305, -3.3552341, 3.3808990
8: -5.0106936, -1.4988346, -5.0634212, -1.4775352, -2.8448887, 2.8740120
9: 0.5460739, 3.0406244, 0.4596386, 3.0538754, -2.3226361, 2.3959835

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847573, upper bound: 1.5897747
time: 4.87 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847573, upper bound: 1.5897746
time: 4.84 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -9.7078648, -5.1545439, -9.6897669, -5.0843935, -3.2558241, 3.2234752
1: -15.0918541, -10.8536530, -15.0866137, -10.8494511, -3.4154081, 3.4089868
2: -9.0469761, -5.7703671, -9.0504360, -5.7714615, -2.8293715, 2.8302057
3: -11.5546618, -7.4121876, -11.5158958, -7.4107313, -3.7833834, 3.7389555
4: -5.4751310, -1.8810148, -5.4693184, -1.9629140, -3.2851601, 3.3110766
5: -3.5717814, -0.4661016, -3.5676656, -0.4975672, -2.7141094, 2.7084756
6: -11.5656462, -6.9906721, -11.5776443, -6.9814620, -3.5602884, 3.5663900
7: -2.7890306, 0.8549109, -2.8021240, 0.8234258, -3.3817291, 3.4477284
8: -5.0350156, -1.4760799, -5.0634089, -1.4775414, -2.8720198, 2.8959231
9: 0.4655366, 3.0619168, 0.4596500, 3.0538676, -2.4052172, 2.4332824

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5913343
time: 4.86 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863623, upper bound: 1.5913326
time: 4.87 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648413, -9.6942806, -5.0616770, -3.3064113, 3.2449231
1: -15.0922251, -10.8474188, -15.0952425, -10.8431473, -3.4361439, 3.4356420
2: -9.0594234, -5.7682381, -9.0615978, -5.7651587, -2.8260384, 2.8452427
3: -11.5214596, -7.4080429, -11.5230656, -7.4036608, -3.7760029, 3.7610607
4: -5.4654350, -1.9569894, -5.4777827, -1.9554484, -3.2659268, 3.2711706
5: -3.5687125, -0.4961605, -3.5736499, -0.4953117, -2.6985550, 2.7043228
6: -11.5807705, -6.9714108, -11.5837259, -6.9704914, -3.5898108, 3.5623035
7: -2.8032002, 0.8274660, -2.8098021, 0.8292036, -3.4122338, 3.4188337
8: -5.0741301, -1.4750619, -5.0775828, -1.4732656, -2.8757257, 2.9146743
9: 0.4387255, 3.0539315, 0.4356761, 3.0562549, -2.4394603, 2.4455857

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928197, upper bound: 1.5856887
time: 4.85 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928192, upper bound: 1.5911281
time: 5.22 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.7338896, -5.0528736, -9.6942778, -5.0616894, -3.3477111, 3.2705989
1: -15.1308241, -10.8254194, -15.0952330, -10.8431702, -3.4771099, 3.4577932
2: -9.0964832, -5.7389607, -9.0615883, -5.7651682, -2.8696604, 2.8794777
3: -11.5864315, -7.3764939, -11.5230608, -7.4036779, -3.8365479, 3.7873154
4: -5.5169415, -1.8470781, -5.4777412, -1.9554546, -3.3365374, 3.3438985
5: -3.6025033, -0.4556713, -3.5736322, -0.4953117, -2.7473583, 2.7254512
6: -11.5967197, -6.9421349, -11.5837164, -6.9704962, -3.6086025, 3.5910220
7: -2.8274059, 0.8817334, -2.8097758, 0.8291955, -3.4389243, 3.4842680
8: -5.0984612, -1.4521980, -5.0775695, -1.4732714, -2.9026604, 2.9366927
9: 0.3572474, 3.0751033, 0.4356866, 3.0562463, -2.5230145, 2.4940507

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6139

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5923903, upper bound: 1.5927701
time: 4.99 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928240, upper bound: 1.5928231
time: 5.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.04 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5847573, upper bound: 1.5897747
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5847573, upper bound: 1.5897746
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5913343
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5863623, upper bound: 1.5913326
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5928197, upper bound: 1.5856887
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5928192, upper bound: 1.5911281
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5923903, upper bound: 1.5927701
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 25.04
Output dim: 9, lower bound: -1.5928240, upper bound: 1.5928231

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6663475, -5.1665659, -9.6878738, -5.0875435, -3.2276230, 3.1943207
1: -15.0539322, -10.8756943, -15.0836124, -10.8536959, -3.3705306, 3.3837981
2: -9.0098915, -5.7996831, -9.0483303, -5.7745314, -2.7827067, 2.7941749
3: -11.4896049, -7.4438972, -11.5142956, -7.4150791, -3.7167025, 3.7107239
4: -5.4243779, -1.9902939, -5.4570236, -1.9644450, -3.2121730, 3.2228556
5: -3.5385942, -0.5064659, -3.5627422, -0.4984102, -2.6640654, 2.6806815
6: -11.5496006, -7.0200167, -11.5747032, -6.9823761, -3.5405045, 3.5344353
7: -2.7651610, 0.8011990, -2.7955523, 0.8216958, -3.3516483, 3.3718538
8: -5.0106936, -1.4988346, -5.0599666, -1.4793258, -2.8431988, 2.8708878
9: 0.5460739, 3.0406244, 0.4626765, 3.0515552, -2.3183599, 2.3928518

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847521, upper bound: 1.5842211
time: 5.02 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847518, upper bound: 1.5897690
time: 4.89 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6663475, -5.1665659, -9.7287750, -5.0756660, -3.2407160, 3.2350245
1: -15.0539322, -10.8756943, -15.1217127, -10.8322353, -3.3922462, 3.4242392
2: -9.0098915, -5.7996831, -9.0844049, -5.7457571, -2.8164482, 2.8363955
3: -11.4896049, -7.4438972, -11.5773048, -7.3840237, -3.7425251, 3.7705274
4: -5.4243779, -1.9902939, -5.5077929, -1.8566020, -3.2788477, 3.2725086
5: -3.5385942, -0.5064659, -3.5960193, -0.4582996, -2.6862788, 2.7176199
6: -11.5496006, -7.0200167, -11.5900116, -6.9533157, -3.5644808, 3.5507107
7: -2.7651610, 0.8011990, -2.8189726, 0.8746052, -3.4055266, 3.3976908
8: -5.0106936, -1.4988346, -5.0838513, -1.4567647, -2.8649163, 2.8971882
9: 0.5460739, 3.0406244, 0.3825045, 3.0719845, -2.3427982, 2.4535277

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 471

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847521, upper bound: 1.5842210
time: 4.96 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847518, upper bound: 1.5897690
time: 4.87 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -9.7077608, -5.1545496, -9.6888924, -5.0866570, -3.2535543, 3.2225285
1: -15.0918293, -10.8536844, -15.0840015, -10.8497868, -3.4150133, 3.4055781
2: -9.0469608, -5.7704129, -9.0493603, -5.7725725, -2.8282022, 2.8285778
3: -11.5545511, -7.4122019, -11.5152884, -7.4121037, -3.7814274, 3.7382550
4: -5.4751143, -1.8811771, -5.4643450, -1.9636225, -3.2843924, 3.3057163
5: -3.5717649, -0.4661684, -3.5662308, -0.4984541, -2.7126780, 2.7065809
6: -11.5655794, -6.9906826, -11.5765305, -6.9817667, -3.5592957, 3.5647616
7: -2.7890129, 0.8548770, -2.8012271, 0.8211455, -3.3789911, 3.4467652
8: -5.0349793, -1.4761019, -5.0623045, -1.4782791, -2.8710423, 2.8942964
9: 0.4656043, 3.0619102, 0.4610715, 3.0516431, -2.4029441, 2.4319601

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5888768
time: 6.34 seconds

## Relational analysis of IS_A1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5913343
time: 5.54 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -9.7078571, -5.1545525, -9.7431364, -5.0780787, -3.2654123, 3.2642493
1: -15.0918465, -10.8536530, -15.1173744, -10.8023100, -3.4505520, 3.4592698
2: -9.0469723, -5.7703719, -9.0652580, -5.7391386, -2.8759365, 2.8653643
3: -11.5546598, -7.4121914, -11.5550404, -7.3918891, -3.7987709, 3.7821503
4: -5.4751177, -1.8810186, -5.5025311, -1.8929678, -3.3154087, 3.3495092
5: -3.5717769, -0.4661045, -3.5924997, -0.4571657, -2.7290068, 2.7285633
6: -11.5656395, -6.9906740, -11.5909252, -6.9599953, -3.5800114, 3.5968785
7: -2.7890291, 0.8549008, -2.8543570, 0.8353348, -3.4140182, 3.4686592
8: -5.0350127, -1.4760828, -5.0922484, -1.4525480, -2.9046245, 2.9172652
9: 0.4655428, 3.0619116, 0.4089417, 3.0645401, -2.4175129, 2.4561052

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 485

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5808349, upper bound: 1.5913269
time: 4.48 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863567, upper bound: 1.5913269
time: 4.98 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -9.6712322, -5.1520877, -9.6922588, -5.0753369, -3.2455068, 3.1796839
1: -15.0553341, -10.9664803, -15.0916109, -10.8525219, -3.3345280, 3.3021979
2: -9.0435333, -5.8407331, -9.0597706, -5.7709217, -2.8032246, 2.7648826
3: -11.5050220, -7.4456034, -11.5209293, -7.4105458, -3.7497773, 3.7246742
4: -5.4216862, -2.0318365, -5.4729757, -1.9622560, -3.2028103, 3.1876163
5: -3.4384532, -0.5122719, -3.5632348, -0.4999552, -2.5370364, 2.5923896
6: -11.4884567, -6.9903512, -11.5763254, -6.9749527, -3.4795227, 3.4949608
7: -2.7756500, 0.8098340, -2.8060474, 0.8265777, -3.3808699, 3.3892155
8: -5.0552340, -1.5594053, -5.0738683, -1.4801812, -2.7947273, 2.7712510
9: 0.4704814, 2.9882627, 0.4390392, 3.0510650, -2.3685694, 2.3592787

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_A2_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911297, upper bound: 1.5856892
time: 5.00 seconds

## Relational analysis of IS_A2_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911297, upper bound: 1.5856892
time: 4.82 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -9.6923933, -5.0648437, -9.6942806, -5.0616770, -3.3064103, 3.2110612
1: -15.0922241, -10.8474178, -15.0952425, -10.8431473, -3.4361448, 3.4073863
2: -9.0594244, -5.7682376, -9.0615978, -5.7651587, -2.8260384, 2.8240068
3: -11.5214596, -7.4080424, -11.5230656, -7.4036608, -3.7760038, 3.7724242
4: -5.4654360, -1.9569901, -5.4777827, -1.9554484, -3.2659278, 3.2428260
5: -3.5687113, -0.4961615, -3.5736499, -0.4953117, -2.6743889, 2.7043226
6: -11.5807724, -6.9714093, -11.5837259, -6.9704914, -3.5898099, 3.5786228
7: -2.8032002, 0.8274655, -2.8098021, 0.8292036, -3.4122319, 3.4344826
8: -5.0741301, -1.4750619, -5.0775828, -1.4732656, -2.8972430, 2.9146726
9: 0.4387255, 3.0539317, 0.4356761, 3.0562549, -2.4394608, 2.4273558

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_A1_A2_A1

### Relational analysis result of IS_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928182, upper bound: 1.5903330
time: 4.77 seconds

## Relational analysis of IS_A2_A1_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928180, upper bound: 1.5911269
time: 4.80 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.6969795, -5.0650907, -9.6852856, -5.0642991, -3.3076806, 3.2488081
1: -15.1031418, -10.8670111, -15.0893192, -10.8531961, -3.4370894, 3.4087636
2: -9.0546074, -5.7566633, -9.0514193, -5.7688494, -2.8249903, 2.8478014
3: -11.5672073, -7.3904157, -11.5183754, -7.4066467, -3.8131800, 3.7632999
4: -5.4755564, -1.8568380, -5.4679503, -1.9574789, -3.2924290, 3.3171992
5: -3.5826335, -0.4697685, -3.5693374, -0.4986925, -2.7201877, 2.7076144
6: -11.5727358, -6.9533644, -11.5780277, -6.9726534, -3.5836744, 3.5700030
7: -2.7823668, 0.8707628, -2.7989902, 0.8269544, -3.3850393, 3.4464266
8: -5.0887647, -1.4712691, -5.0754185, -1.4778357, -2.8882866, 2.9160857
9: 0.3824511, 3.0683615, 0.4414039, 3.0547957, -2.4950860, 2.4807990

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5923852, upper bound: 1.5873124
time: 4.98 seconds

## Relational analysis of IS_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5923849, upper bound: 1.5927646
time: 5.16 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.7338390, -5.0528874, -9.6942778, -5.0616894, -3.3347859, 3.2705741
1: -15.1307774, -10.8254976, -15.0952330, -10.8431702, -3.4770460, 3.4499743
2: -9.0963020, -5.7390223, -9.0615883, -5.7651682, -2.8559747, 2.8787792
3: -11.5861444, -7.3765802, -11.5230608, -7.4036779, -3.8483233, 3.7865200
4: -5.5168462, -1.8473263, -5.4777412, -1.9554546, -3.3332520, 3.3432410
5: -3.6024430, -0.4556961, -3.5736322, -0.4953117, -2.7472897, 2.7192078
6: -11.5966377, -6.9421706, -11.5837164, -6.9704962, -3.6102061, 3.5905638
7: -2.8272867, 0.8815293, -2.8097758, 0.8291955, -3.4387898, 3.4892166
8: -5.0984006, -1.4522357, -5.0775695, -1.4732714, -2.9024220, 2.9349015
9: 0.3574095, 3.0749698, 0.4356866, 3.0562463, -2.5223112, 2.5028107

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_A2_A2_A2_A1

### Relational analysis result of IS_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928191, upper bound: 1.5873748
time: 4.85 seconds

## Relational analysis of IS_A2_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928186, upper bound: 1.5928172
time: 5.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.74 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5847521, upper bound: 1.5842211
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5847518, upper bound: 1.5897690
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5847521, upper bound: 1.5842210
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5847518, upper bound: 1.5897690
IS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5888768
IS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5838354, upper bound: 1.5913343
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5808349, upper bound: 1.5913269
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5863567, upper bound: 1.5913269
IS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5911297, upper bound: 1.5856892
IS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5911297, upper bound: 1.5856892
IS_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5928182, upper bound: 1.5903330
IS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5928180, upper bound: 1.5911269
IS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5923852, upper bound: 1.5873124
IS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5923849, upper bound: 1.5927646
IS_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5928191, upper bound: 1.5873748
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.74
Output dim: 9, lower bound: -1.5928186, upper bound: 1.5928172

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -9.6449871, -5.2536902, -9.6858511, -5.1012034, -3.1434817, 3.1292253
1: -15.0170202, -10.9946852, -15.0799789, -10.8630676, -3.2641706, 3.2502384
2: -8.9940023, -5.8721385, -9.0464993, -5.7802935, -2.7608070, 2.7141979
3: -11.4731760, -7.4818916, -11.5121574, -7.4219913, -3.6905508, 3.6740646
4: -5.3806863, -2.0649452, -5.4522114, -1.9712486, -3.1447964, 3.1393223
5: -3.4084206, -0.5225811, -3.5523269, -0.5030546, -2.5026464, 2.5694304
6: -11.4573135, -7.0389557, -11.5673046, -6.9868383, -3.4300265, 3.4674735
7: -2.7377887, 0.7836280, -2.7917972, 0.8190722, -3.3206491, 3.3424392
8: -4.9921775, -1.5831776, -5.0562510, -1.4862409, -2.7627597, 2.7275150
9: 0.5780840, 2.9749885, 0.4660459, 3.0463645, -2.2472186, 2.3065393

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_A1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847346, upper bound: 1.5817508
time: 4.90 seconds

## Relational analysis of IS_A1_A1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847346, upper bound: 1.5842041
time: 4.94 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -9.6663485, -5.1665683, -9.6878738, -5.0875435, -3.2269459, 3.1604586
1: -15.0539322, -10.8756943, -15.0836124, -10.8536959, -3.3705306, 3.3555429
2: -9.0098915, -5.7996840, -9.0483303, -5.7745314, -2.7827077, 2.7729397
3: -11.4896059, -7.4438982, -11.5142956, -7.4150791, -3.7167034, 3.7220883
4: -5.4243784, -1.9902941, -5.4570236, -1.9644450, -3.2121730, 3.1945090
5: -3.5385938, -0.5064659, -3.5627422, -0.4984102, -2.6399021, 2.6806824
6: -11.5496006, -7.0200157, -11.5747032, -6.9823761, -3.5405045, 3.5507565
7: -2.7651620, 0.8011975, -2.7955523, 0.8216958, -3.3516474, 3.3875651
8: -5.0106921, -1.4988356, -5.0599666, -1.4793258, -2.8647175, 2.8708873
9: 0.5460744, 3.0406232, 0.4626765, 3.0515552, -2.3183608, 2.3746226

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A1_A1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847509, upper bound: 1.5891067
time: 5.21 seconds

## Relational analysis of IS_A1_A1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5847508, upper bound: 1.5897680
time: 4.93 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -9.6449871, -5.2536902, -9.7267494, -5.0893345, -3.1566906, 3.1703649
1: -15.0170202, -10.9946852, -15.1180859, -10.8415966, -3.2857971, 3.2905014
2: -8.9940023, -5.8721385, -9.0825233, -5.7515202, -2.7944460, 2.7564340
3: -11.4731760, -7.4818916, -11.5751629, -7.3909054, -3.7163777, 3.7336130
4: -5.3806863, -2.0649452, -5.5029821, -1.8634092, -3.1838360, 3.1889591
5: -3.4084206, -0.5225811, -3.5856142, -0.4629402, -2.5247355, 2.6070068
6: -11.4573135, -7.0389557, -11.5826092, -6.9577718, -3.4543943, 3.4835649
7: -2.7377887, 0.7836280, -2.8152318, 0.8719773, -3.3746281, 3.3675218
8: -4.9921775, -1.5831776, -5.0801649, -1.4636712, -2.7756281, 2.7544830
9: 0.5780840, 2.9749885, 0.3858471, 3.0667727, -2.2718389, 2.3681109

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863576, upper bound: 1.5817505
time: 5.12 seconds

## Relational analysis of IS_A1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863575, upper bound: 1.5842037
time: 4.94 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -9.6663485, -5.1665683, -9.7287750, -5.0756660, -3.2400389, 3.2011642
1: -15.0539322, -10.8756943, -15.1217127, -10.8322353, -3.3922472, 3.3959842
2: -9.0098915, -5.7996840, -9.0844049, -5.7457571, -2.8164482, 2.8151598
3: -11.4896059, -7.4438982, -11.5773048, -7.3840237, -3.7425270, 3.7816887
4: -5.4243784, -1.9902941, -5.5077929, -1.8566020, -3.2782297, 3.2441621
5: -3.5385938, -0.5064659, -3.5960193, -0.4582996, -2.6621065, 2.7176213
6: -11.5496006, -7.0200157, -11.5900116, -6.9533157, -3.5644798, 3.5670319
7: -2.7651620, 0.8011975, -2.8189726, 0.8746052, -3.4055266, 3.4134021
8: -5.0106921, -1.4988356, -5.0838513, -1.4567647, -2.8864336, 2.8971877
9: 0.5460744, 3.0406232, 0.3825045, 3.0719845, -2.3427992, 2.4352961

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A1_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863734, upper bound: 1.5891064
time: 4.99 seconds

## Relational analysis of IS_A1_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863733, upper bound: 1.5897683
time: 4.91 seconds

## BFS IS instance: IS_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.7068739, -5.1568022, -9.6888924, -5.0866570, -3.2527008, 3.2203333
1: -15.0892315, -10.8540287, -15.0840015, -10.8497868, -3.4116507, 3.4052110
2: -9.0459480, -5.7715449, -9.0493603, -5.7725725, -2.8266439, 2.8274026
3: -11.5539360, -7.4135685, -11.5152884, -7.4121037, -3.7807493, 3.7360997
4: -5.4701757, -1.8818797, -5.4643450, -1.9636225, -3.2788267, 3.3049829
5: -3.5702665, -0.4670506, -3.5662308, -0.4984541, -2.7109432, 2.7051589
6: -11.5644703, -6.9909887, -11.5765305, -6.9817667, -3.5581589, 3.5638738
7: -2.7881160, 0.8526187, -2.8012271, 0.8211455, -3.3780680, 3.4440303
8: -5.0338850, -1.4768391, -5.0623045, -1.4782791, -2.8694496, 2.8933806
9: 0.4670348, 3.0597293, 0.4610715, 3.0516431, -2.4016142, 2.4297791

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 485

## Relational analysis of IS_A1_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5783440, upper bound: 1.5888712
time: 4.65 seconds

## Relational analysis of IS_A1_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838298, upper bound: 1.5888712
time: 5.89 seconds

## BFS IS instance: IS_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.7605486, -5.1485863, -9.6888924, -5.0866570, -3.2773838, 3.2303369
1: -15.1202106, -10.8067989, -15.0840015, -10.8497868, -3.4451828, 3.4531856
2: -9.0632162, -5.7388654, -9.0493603, -5.7725725, -2.8527784, 2.8658941
3: -11.5940046, -7.3944397, -11.5152884, -7.4121037, -3.8013763, 3.7547059
4: -5.5067225, -1.8116196, -5.4643450, -1.9636225, -3.3189688, 3.3257654
5: -3.5963011, -0.4289856, -3.5662308, -0.4984541, -2.7360396, 2.7157426
6: -11.5780201, -6.9691668, -11.5765305, -6.9817667, -3.5806365, 3.5919178
7: -2.8410077, 0.8662338, -2.8012271, 0.8211455, -3.4331808, 3.4586184
8: -5.0615749, -1.4515409, -5.0623045, -1.4782791, -2.9123650, 2.9095807
9: 0.4164648, 3.0724943, 0.4610715, 3.0516431, -2.4247570, 2.4425986

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6139

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_A1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838302, upper bound: 1.5857793
time: 4.94 seconds

## Relational analysis of IS_A1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5838299, upper bound: 1.5913286
time: 4.72 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.7058210, -5.1682110, -9.7217598, -5.1655626, -3.2015257, 3.1799448
1: -15.0882111, -10.8630095, -15.0787859, -10.9214306, -3.3157640, 3.3339653
2: -9.0450974, -5.7761307, -9.0486050, -5.8116016, -2.7962508, 2.8066671
3: -11.5525188, -7.4191194, -11.5388241, -7.4302540, -3.7617249, 3.7560315
4: -5.4703999, -1.8878160, -5.4583988, -1.9681247, -3.2322569, 3.2537127
5: -3.5613713, -0.4707479, -3.4623470, -0.4742670, -2.6055202, 2.5667295
6: -11.5582438, -6.9951315, -11.4981976, -6.9790936, -3.4860725, 3.4852910
7: -2.7852798, 0.8522782, -2.8270040, 0.8175168, -3.3828964, 3.4354661
8: -5.0313282, -1.4829884, -5.0718288, -1.5366898, -2.7621312, 2.7984219
9: 0.4689069, 3.0567017, 0.4412637, 2.9989133, -2.3320467, 2.3563492

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5801577, upper bound: 1.5913256
time: 5.50 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5808338, upper bound: 1.5913259
time: 4.65 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.7078571, -5.1545525, -9.7428637, -5.0781965, -3.2312279, 3.2632685
1: -15.0918465, -10.8536530, -15.1155052, -10.8023138, -3.4222817, 3.4571528
2: -9.0469723, -5.7703719, -9.0651188, -5.7393985, -2.8541098, 2.8645418
3: -11.5546598, -7.4121914, -11.5549679, -7.3922443, -3.8095579, 3.7818718
4: -5.4751177, -1.8810186, -5.5020447, -1.8930289, -3.2869983, 3.3481262
5: -3.5717769, -0.4661045, -3.5922923, -0.4591799, -2.7266159, 2.7042756
6: -11.5656395, -6.9906740, -11.5903587, -6.9601393, -3.5947104, 3.5953474
7: -2.7890291, 0.8549008, -2.8541217, 0.8349833, -3.4280553, 3.4684339
8: -5.0350127, -1.4760828, -5.0908813, -1.4527702, -2.9040623, 2.9371774
9: 0.4655428, 3.0619116, 0.4098506, 3.0645187, -2.3992629, 2.4549513

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6155

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5856639, upper bound: 1.5913263
time: 4.58 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5863556, upper bound: 1.5913259
time: 4.98 seconds

## BFS IS instance: IS_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6712322, -5.1520877, -9.6903725, -5.0784998, -3.2407823, 3.1763084
1: -15.0553341, -10.9664803, -15.0885944, -10.8567886, -3.3299932, 3.2990913
2: -9.0435333, -5.8407331, -9.0576000, -5.7740002, -2.8002095, 2.7633722
3: -11.5050220, -7.4456034, -11.5193253, -7.4149246, -3.7449808, 3.7226758
4: -5.4216862, -2.0318365, -5.4606304, -1.9637941, -3.2005062, 3.1731067
5: -3.4384532, -0.5122719, -3.5582972, -0.5008020, -2.5360022, 2.5865004
6: -11.4884567, -6.9903512, -11.5733738, -6.9758663, -3.4784513, 3.4916248
7: -2.7756500, 0.8098340, -2.7994518, 0.8248415, -3.3772907, 3.3802347
8: -5.0552340, -1.5594053, -5.0704179, -1.4819775, -2.7930956, 2.7681406
9: 0.4704814, 2.9882627, 0.4420838, 3.0487413, -2.3642869, 2.3561239

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6139

## Relational analysis of IS_A2_A1_A1_B1_A1

### Relational analysis result of IS_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5906912, upper bound: 1.5856198
time: 5.01 seconds

## Relational analysis of IS_A2_A1_A1_B1_A2

### Relational analysis result of IS_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911291, upper bound: 1.5856885
time: 4.96 seconds

## BFS IS instance: IS_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6712322, -5.1520877, -9.7312622, -5.0666370, -3.2539892, 3.2174456
1: -15.0553341, -10.9664803, -15.1268368, -10.8353167, -3.3515668, 3.3394482
2: -9.0435333, -5.8407331, -9.0935631, -5.7452211, -2.8337369, 2.8056023
3: -11.5050220, -7.4456034, -11.5823145, -7.3838835, -3.7707291, 3.7814441
4: -5.4216862, -2.0318365, -5.5115595, -1.8558248, -3.2401171, 3.2218881
5: -3.4384532, -0.5122719, -3.5917153, -0.4606609, -2.5561638, 2.6240711
6: -11.4884567, -6.9903512, -11.5886612, -6.9468169, -3.5070601, 3.5077097
7: -2.7756500, 0.8098340, -2.8229489, 0.8778582, -3.4312539, 3.4051499
8: -5.0552340, -1.5594053, -5.0943174, -1.4593830, -2.8060336, 2.7950554
9: 0.4704814, 2.9882627, 0.3617043, 3.0691261, -2.3889978, 2.4380565

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_A1_A1_B2_A1

### Relational analysis result of IS_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911284, upper bound: 1.5848827
time: 4.73 seconds

## Relational analysis of IS_A2_A1_A1_B2_A2

### Relational analysis result of IS_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5911284, upper bound: 1.5856881
time: 5.59 seconds

## BFS IS instance: IS_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.6512356, -5.0780897, -9.6914845, -5.0624743, -3.2749043, 3.1969676
1: -15.0474968, -10.8663874, -15.0919971, -10.8444023, -3.3771429, 3.3790574
2: -9.0413675, -5.8187485, -9.0604439, -5.7685094, -2.8020186, 2.7795033
3: -11.5053825, -7.4457655, -11.5216484, -7.4060907, -3.7566590, 3.7219176
4: -5.4229603, -1.9768759, -5.4749537, -1.9570576, -3.2227974, 3.2233672
5: -3.5105958, -0.5057306, -3.5697198, -0.4961929, -2.6104994, 2.6613207
6: -11.5134363, -6.9838171, -11.5792141, -6.9715524, -3.5173492, 3.5305429
7: -2.7644582, 0.8162532, -2.8072300, 0.8280239, -3.3718572, 3.3984342
8: -5.0523324, -1.5129466, -5.0754495, -1.4757977, -2.7989945, 2.8732531
9: 0.4601135, 3.0372555, 0.4370975, 3.0551364, -2.3986671, 2.4080098

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_A1_A2_A1_A1

### Relational analysis result of IS_A2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928017, upper bound: 1.5877914
time: 12.12 seconds

## Relational analysis of IS_A2_A1_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928017, upper bound: 1.5903187
time: 9.93 seconds

## BFS IS instance: IS_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -9.6923904, -5.0648417, -9.6942806, -5.0616770, -3.3023448, 3.2110600
1: -15.0922222, -10.8474178, -15.0952425, -10.8431473, -3.4374418, 3.4073863
2: -9.0594234, -5.7682390, -9.0615978, -5.7651587, -2.8260384, 2.8221092
3: -11.5214605, -7.4080462, -11.5230656, -7.4036608, -3.7946720, 3.7722201
4: -5.4654350, -1.9569906, -5.4777827, -1.9554484, -3.2633414, 3.2427583
5: -3.5687113, -0.4961615, -3.5736499, -0.4953117, -2.6743879, 2.7302871
6: -11.5807724, -6.9714098, -11.5837259, -6.9704914, -3.5764036, 3.5786228
7: -2.8031993, 0.8274646, -2.8098021, 0.8292036, -3.4122338, 3.4423618
8: -5.0741305, -1.4750624, -5.0775828, -1.4732656, -2.9435601, 2.9146719
9: 0.4387264, 3.0539308, 0.4356761, 3.0562549, -2.4394598, 2.4261134

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_A1_A2_A2_A1

### Relational analysis result of IS_A2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928015, upper bound: 1.5885816
time: 10.81 seconds

## Relational analysis of IS_A2_A1_A2_A2_A2

### Relational analysis result of IS_A2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928015, upper bound: 1.5911122
time: 10.93 seconds

## BFS IS instance: IS_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -9.6758003, -5.1523857, -9.6832638, -5.0779610, -3.2280598, 3.1836729
1: -15.0668736, -10.9859238, -15.0856953, -10.8625679, -3.3009558, 3.2744522
2: -9.0380650, -5.8290696, -9.0495930, -5.7746119, -2.7884035, 2.7680473
3: -11.5507069, -7.4272394, -11.5162373, -7.4135361, -3.7878580, 3.7274404
4: -5.4318404, -1.9317136, -5.4631410, -1.9642848, -3.2108836, 3.2341316
5: -3.4528339, -0.4858656, -3.5589232, -0.5033350, -2.5587378, 2.5833690
6: -11.4802799, -6.9722390, -11.5706234, -6.9771099, -3.4729452, 3.4889386
7: -2.7544990, 0.8531923, -2.7952313, 0.8243308, -3.3524432, 3.4157324
8: -5.0699134, -1.5555863, -5.0717044, -1.4847555, -2.8045573, 2.7731054
9: 0.4139910, 3.0024617, 0.4447694, 3.0496068, -2.3959343, 2.3944042

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_A2_A2_A1_A1_A1

### Relational analysis result of IS_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5923688, upper bound: 1.5847990
time: 4.97 seconds

## Relational analysis of IS_A2_A2_A1_A1_A2

### Relational analysis result of IS_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5923688, upper bound: 1.5872958
time: 5.19 seconds

## BFS IS instance: IS_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -9.6969795, -5.0650926, -9.6852856, -5.0642991, -3.3076806, 3.2149475
1: -15.1031408, -10.8670082, -15.0893192, -10.8531961, -3.4368668, 3.3805077
2: -9.0546055, -5.7566619, -9.0514193, -5.7688494, -2.8249893, 2.8265667
3: -11.5672054, -7.3904157, -11.5183754, -7.4066467, -3.8131800, 3.7746630
4: -5.4755564, -1.8568386, -5.4679503, -1.9574789, -3.2924290, 3.2888398
5: -3.5826325, -0.4697685, -3.5693374, -0.4986925, -2.6960154, 2.7059362
6: -11.5727329, -6.9533629, -11.5780277, -6.9726534, -3.5836725, 3.5863228
7: -2.7823672, 0.8707633, -2.7989902, 0.8269544, -3.3850393, 3.4619789
8: -5.0887642, -1.4712682, -5.0754185, -1.4778357, -2.9098063, 2.9160852
9: 0.3824515, 3.0683618, 0.4414039, 3.0547957, -2.4945242, 2.4625680

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_A2_A2_A1_A2_A1

### Relational analysis result of IS_A2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5923838, upper bound: 1.5919517
time: 4.90 seconds

## Relational analysis of IS_A2_A2_A1_A2_A2

### Relational analysis result of IS_A2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5923837, upper bound: 1.5927634
time: 5.09 seconds

## BFS IS instance: IS_A2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.7126656, -5.1402016, -9.6922569, -5.0753522, -3.2529964, 3.2053590
1: -15.0939465, -10.9444027, -15.0915995, -10.8525410, -3.3590221, 3.3159313
2: -9.0797176, -5.8115373, -9.0597610, -5.7709322, -2.8189139, 2.7985799
3: -11.5696373, -7.4138560, -11.5209255, -7.4105639, -3.8230295, 3.7503080
4: -5.4733381, -1.9222015, -5.4729342, -1.9622611, -3.2522559, 3.2601500
5: -3.4722762, -0.4718008, -3.5632186, -0.4999571, -2.5856972, 2.5935445
6: -11.5042915, -6.9610796, -11.5763168, -6.9749575, -3.4996300, 3.5101409
7: -2.7994828, 0.8638701, -2.8060207, 0.8265715, -3.4065189, 3.4582710
8: -5.0794382, -1.5364828, -5.0738554, -1.4801855, -2.8212228, 2.7933471
9: 0.3888159, 3.0090117, 0.4390492, 3.0510571, -2.4230509, 2.4163711

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6208

## Relational analysis of IS_A2_A2_A2_A1_A1

### Relational analysis result of IS_A2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5884057, upper bound: 1.5873710
time: 5.09 seconds

## Relational analysis of IS_A2_A2_A2_A1_A2

### Relational analysis result of IS_A2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.5928160, upper bound: 1.5873703
time: 4.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 24.62 seconds
IS_A1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5847346, upper bound: 1.5817508
IS_A1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5847346, upper bound: 1.5842041
IS_A1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5847509, upper bound: 1.5891067
IS_A1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5847508, upper bound: 1.5897680
IS_A1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5863576, upper bound: 1.5817505
IS_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5863575, upper bound: 1.5842037
IS_A1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5863734, upper bound: 1.5891064
IS_A1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5863733, upper bound: 1.5897683
IS_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5783440, upper bound: 1.5888712
IS_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5838298, upper bound: 1.5888712
IS_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5838302, upper bound: 1.5857793
IS_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5838299, upper bound: 1.5913286
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5801577, upper bound: 1.5913256
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5808338, upper bound: 1.5913259
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5856639, upper bound: 1.5913263
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5863556, upper bound: 1.5913259
IS_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5906912, upper bound: 1.5856198
IS_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5911291, upper bound: 1.5856885
IS_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5911284, upper bound: 1.5848827
IS_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5911284, upper bound: 1.5856881
IS_A2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5928017, upper bound: 1.5877914
IS_A2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5928017, upper bound: 1.5903187
IS_A2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5928015, upper bound: 1.5885816
IS_A2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5928015, upper bound: 1.5911122
IS_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5923688, upper bound: 1.5847990
IS_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5923688, upper bound: 1.5872958
IS_A2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5923838, upper bound: 1.5919517
IS_A2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5923837, upper bound: 1.5927634
IS_A2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5884057, upper bound: 1.5873710
IS_A2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 24.62
Output dim: 9, lower bound: -1.5928160, upper bound: 1.5873703
IS_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.62
Output dim: 9, lower bound: -1.5928186, upper bound: 1.5928172
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.44260835647583
rel_dist={9: [-1.592849651897651, 1.5928489343896768]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6126
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6126

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142331, upper bound: 1.4147649
time: 5.09 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147661, upper bound: 1.4147656
time: 4.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.94 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 9.94
Output dim: 9, lower bound: -1.4142331, upper bound: 1.4147649
IS_B2, status: Status.UNKNOWN, split count: 1, time: 9.94
Output dim: 9, lower bound: -1.4147661, upper bound: 1.4147656

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -9.6931152, -5.0636296, -9.6923923, -5.0648394, -3.0747738, 3.0752976
1: -15.0933781, -10.8457947, -15.0922241, -10.8474169, -3.2117014, 3.2122726
2: -9.0602570, -5.7670708, -9.0594254, -5.7682366, -2.7184396, 2.7190056
3: -11.5220776, -7.4063749, -11.5214615, -7.4080415, -3.5725155, 3.5735650
4: -5.4701414, -1.9563961, -5.4654369, -1.9569887, -3.1587424, 3.1540670
5: -3.5705919, -0.4958344, -3.5687127, -0.4961605, -2.5504065, 2.5485415
6: -11.5818977, -6.9710603, -11.5807753, -6.9714108, -3.3716760, 3.3708577
7: -2.8057141, 0.8281331, -2.8032007, 0.8274651, -3.3017197, 3.2996454
8: -5.0754433, -1.4743752, -5.0741324, -1.4750624, -2.7294569, 2.7289259
9: 0.4375563, 3.0548153, 0.4387226, 3.0539322, -2.3528562, 2.3532896

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4104695, upper bound: 1.4127479
time: 4.58 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142311, upper bound: 1.4147631
time: 4.58 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -9.6942730, -5.0616975, -9.7316389, -5.0531878, -3.0993123, 3.1173232
1: -15.0952234, -10.8431835, -15.1295862, -10.8272877, -3.2338014, 3.2544734
2: -9.0615826, -5.7651758, -9.0930634, -5.7406940, -2.7516017, 2.7597518
3: -11.5230589, -7.4036922, -11.5797920, -7.3782063, -3.5986400, 3.6302009
4: -5.4777136, -1.9554579, -5.5150671, -1.8537853, -3.2336779, 3.2182345
5: -3.5736222, -0.4953146, -3.6012430, -0.4570122, -2.5719490, 2.5938678
6: -11.5837116, -6.9704976, -11.5943975, -6.9428940, -3.4009767, 3.3858166
7: -2.8097610, 0.8291945, -2.8250456, 0.8776355, -3.3661537, 3.3258495
8: -5.0775614, -1.4732752, -5.0968332, -1.4531322, -2.7524137, 2.7534308
9: 0.4356918, 3.0562398, 0.3610535, 3.0728159, -2.3966188, 2.4315884

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6165
type: B, layer: 1, pos: 6165
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6126

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6165

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4109928, upper bound: 1.4127479
time: 4.96 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147641, upper bound: 1.4147647
time: 4.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 24.55 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 24.55
Output dim: 9, lower bound: -1.4104695, upper bound: 1.4127479
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 24.55
Output dim: 9, lower bound: -1.4142311, upper bound: 1.4147631
IS_B2_A1, status: Status.VERIFIED, split count: 2, time: 24.55
Output dim: 9, lower bound: -1.4109928, upper bound: 1.4127479
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 24.55
Output dim: 9, lower bound: -1.4147641, upper bound: 1.4147647

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -9.6931152, -5.0636349, -9.6923943, -5.0648370, -3.0747743, 3.0094368
1: -15.0933771, -10.8457985, -15.0922260, -10.8474178, -3.2116976, 3.2126002
2: -9.0602570, -5.7670708, -9.0594254, -5.7682371, -2.6942544, 2.7185583
3: -11.5220757, -7.4063759, -11.5214615, -7.4080424, -3.5804806, 3.5729132
4: -5.4701400, -1.9563977, -5.4654369, -1.9569886, -3.1586094, 3.1448865
5: -3.5705907, -0.4958334, -3.5687129, -0.4961605, -2.5503850, 2.5485215
6: -11.5818977, -6.9710617, -11.5807753, -6.9714088, -3.3716750, 3.3374844
7: -2.8057117, 0.8281317, -2.8032007, 0.8274670, -3.3013139, 3.2980409
8: -5.0754414, -1.4743762, -5.0741339, -1.4750624, -2.6875315, 2.7287767
9: 0.4375601, 3.0548153, 0.4387217, 3.0539322, -2.3528411, 2.3599806

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6126
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6126

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142311, upper bound: 1.4142302
time: 4.61 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142311, upper bound: 1.4147631
time: 4.66 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -9.6942711, -5.0617013, -9.7316399, -5.0531874, -3.0993099, 3.0514615
1: -15.0952225, -10.8431854, -15.1295853, -10.8272858, -3.2337985, 3.2548008
2: -9.0615816, -5.7651777, -9.0930614, -5.7406940, -2.7273664, 2.7593055
3: -11.5230589, -7.4036913, -11.5797920, -7.3782058, -3.6064968, 3.6283388
4: -5.4777126, -1.9554578, -5.5150676, -1.8537846, -3.2325850, 3.2080717
5: -3.5736222, -0.4953136, -3.6012421, -0.4570112, -2.5709233, 2.5938482
6: -11.5837097, -6.9705000, -11.5943956, -6.9428940, -3.4009738, 3.3524461
7: -2.8097577, 0.8291922, -2.8250465, 0.8776369, -3.3649120, 3.3240643
8: -5.0775595, -1.4732771, -5.0968323, -1.4531312, -2.7105718, 2.7532835
9: 0.4356966, 3.0562408, 0.3610530, 3.0728159, -2.3966026, 2.4370887

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147628, upper bound: 1.4128898
time: 4.71 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147626, upper bound: 1.4147632
time: 4.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 24.28 seconds
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4142311, upper bound: 1.4142302
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4142311, upper bound: 1.4147631
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4147628, upper bound: 1.4128898
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 9, lower bound: -1.4147626, upper bound: 1.4147632

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -9.6923923, -5.0648413, -9.6923943, -5.0648370, -3.0734792, 3.0076196
1: -15.0922251, -10.8474188, -15.0922260, -10.8474178, -3.2105465, 3.2108765
2: -9.0594234, -5.7682381, -9.0594254, -5.7682371, -2.6936631, 2.7174125
3: -11.5214596, -7.4080429, -11.5214615, -7.4080424, -3.5797033, 3.5710897
4: -5.4654350, -1.9569894, -5.4654369, -1.9569886, -3.1530609, 3.1440053
5: -3.5687125, -0.4961605, -3.5687129, -0.4961605, -2.5481277, 2.5481269
6: -11.5807705, -6.9714108, -11.5807753, -6.9714088, -3.3704429, 3.3370709
7: -2.8032002, 0.8274660, -2.8032007, 0.8274670, -3.2978597, 3.2966413
8: -5.0741301, -1.4750619, -5.0741339, -1.4750624, -2.6863499, 2.7281296
9: 0.4387255, 3.0539315, 0.4387217, 3.0539322, -2.3516386, 2.3583446

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6139

## Relational analysis of IS_B1_A2_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4130499, upper bound: 1.4132626
time: 4.49 seconds

## Relational analysis of IS_B1_A2_A1_A2

### Relational analysis result of IS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142307, upper bound: 1.4142298
time: 4.60 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -9.7309351, -5.0533061, -9.6923943, -5.0648370, -3.1118999, 3.0199430
1: -15.1291599, -10.8279886, -15.0922260, -10.8474178, -3.2494583, 3.2309618
2: -9.0916929, -5.7412748, -9.0594254, -5.7682371, -2.7306104, 2.7490215
3: -11.5774574, -7.3788338, -11.5214615, -7.4080424, -3.6301947, 3.5954704
4: -5.5143995, -1.8560826, -5.4654369, -1.9569886, -3.2011490, 3.2073078
5: -3.6008143, -0.4574356, -3.5687129, -0.4961605, -2.5836110, 2.5651274
6: -11.5935764, -6.9431725, -11.5807753, -6.9714088, -3.3838172, 3.3640275
7: -2.8241873, 0.8761139, -2.8032007, 0.8274670, -3.3209043, 3.3452449
8: -5.0962100, -1.4534526, -5.0741339, -1.4750624, -2.7084455, 2.7488170
9: 0.3623996, 3.0720315, 0.4387217, 3.0539322, -2.4257302, 2.3794022

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6139

## Relational analysis of IS_B1_A2_A2_A1

### Relational analysis result of IS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4130499, upper bound: 1.4138006
time: 4.94 seconds

## Relational analysis of IS_B1_A2_A2_A2

### Relational analysis result of IS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142307, upper bound: 1.4147635
time: 4.81 seconds

## BFS IS instance: IS_B2_A2_A1

### Backsubstitution after applying IS history:
0: -9.6731129, -5.1489725, -9.7267923, -5.0808096, -3.0147767, 2.9831934
1: -15.0583372, -10.9622250, -15.1211557, -10.8533688, -3.1145573, 3.1132660
2: -9.0456848, -5.8376837, -9.0888977, -5.7566757, -2.6964407, 2.6763182
3: -11.5066042, -7.4412680, -11.5753651, -7.3910532, -3.5765400, 3.5891175
4: -5.4339738, -2.0303168, -5.5042253, -1.8710167, -3.1304564, 3.1186929
5: -3.4433618, -0.5114322, -3.5724697, -0.4640751, -2.4092960, 2.4591610
6: -11.4913874, -6.9894543, -11.5739689, -6.9501324, -3.2888193, 3.2707875
7: -2.7821918, 0.8115406, -2.8170233, 0.8726616, -3.3270855, 3.2898321
8: -5.0586843, -1.5576067, -5.0906062, -1.4719887, -2.6079426, 2.6114931
9: 0.4674602, 2.9905884, 0.3687854, 3.0583224, -2.2975597, 2.3469737

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147628, upper bound: 1.4116656
time: 6.79 seconds

## Relational analysis of IS_B2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147623, upper bound: 1.4128921
time: 8.15 seconds

## BFS IS instance: IS_B2_A2_A2

### Backsubstitution after applying IS history:
0: -9.6942720, -5.0617027, -9.7316399, -5.0531874, -3.0993109, 3.0143735
1: -15.0952215, -10.8431873, -15.1295853, -10.8272858, -3.2337966, 3.2238545
2: -9.0615807, -5.7651777, -9.0930614, -5.7406940, -2.7273674, 2.7360487
3: -11.5230589, -7.4036932, -11.5797920, -7.3782058, -3.6064968, 3.6386514
4: -5.4777112, -1.9554586, -5.5150676, -1.8537846, -3.2319717, 3.1770258
5: -3.5736203, -0.4953156, -3.6012421, -0.4570112, -2.5444460, 2.5925455
6: -11.5837078, -6.9705024, -11.5943956, -6.9428940, -3.4009738, 3.3675447
7: -2.8097582, 0.8291907, -2.8250465, 0.8776369, -3.3649111, 3.3382950
8: -5.0775576, -1.4732780, -5.0968323, -1.4531312, -2.7304821, 2.7532825
9: 0.4356976, 3.0562396, 0.3610530, 3.0728159, -2.3966026, 2.4171185

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147622, upper bound: 1.4144255
time: 4.82 seconds

## Relational analysis of IS_B2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147621, upper bound: 1.4147622
time: 4.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 24.18 seconds
IS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4130499, upper bound: 1.4132626
IS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4142307, upper bound: 1.4142298
IS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4130499, upper bound: 1.4138006
IS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4142307, upper bound: 1.4147635
IS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4147628, upper bound: 1.4116656
IS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4147623, upper bound: 1.4128921
IS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4147622, upper bound: 1.4144255
IS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 24.18
Output dim: 9, lower bound: -1.4147621, upper bound: 1.4147622

## BFS IS instance: IS_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -9.6547890, -5.0769010, -9.6733446, -5.0703845, -3.0294509, 2.9775259
1: -15.0645370, -10.8892689, -15.0795116, -10.8686848, -3.1589570, 3.1580973
2: -9.0170031, -5.7857351, -9.0378990, -5.7760863, -2.6437960, 2.6720097
3: -11.5016813, -7.4215131, -11.5115681, -7.4143257, -3.5523844, 3.5426407
4: -5.4243231, -1.9664235, -5.4447579, -1.9613178, -3.1063442, 3.1126828
5: -3.5490360, -0.5102720, -3.5595188, -0.5033283, -2.5209560, 2.5252168
6: -11.5567665, -6.9825110, -11.5687103, -6.9760323, -3.3357892, 3.3112659
7: -2.7580988, 0.8165779, -2.7804089, 0.8226824, -3.2414007, 3.2580175
8: -5.0646863, -1.4944286, -5.0695553, -1.4847121, -2.6667938, 2.7047462
9: 0.4636350, 3.0470600, 0.4508972, 3.0508280, -2.3219614, 2.3385932

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_B1_A2_A1_A1_A1

### Relational analysis result of IS_B1_A2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4130486, upper bound: 1.4113864
time: 4.40 seconds

## Relational analysis of IS_B1_A2_A1_A1_A2

### Relational analysis result of IS_B1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4130484, upper bound: 1.4132611
time: 4.40 seconds

## BFS IS instance: IS_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -9.6923895, -5.0648437, -9.6923943, -5.0648384, -3.0593815, 3.0076199
1: -15.0922222, -10.8474207, -15.0922260, -10.8474159, -3.2105455, 3.2023621
2: -9.0594225, -5.7682381, -9.0594254, -5.7682366, -2.6760006, 2.7167583
3: -11.5214577, -7.4080439, -11.5214596, -7.4080429, -3.5897355, 3.5703344
4: -5.4654331, -1.9569905, -5.4654369, -1.9569877, -3.1490140, 3.1437445
5: -3.5687113, -0.4961624, -3.5687127, -0.4961605, -2.5481277, 2.5413260
6: -11.5807714, -6.9714108, -11.5807743, -6.9714098, -3.3689861, 3.3366222
7: -2.8031983, 0.8274641, -2.8032002, 0.8274655, -3.2978201, 3.3026476
8: -5.0741296, -1.4750638, -5.0741334, -1.4750609, -2.6862373, 2.7257366
9: 0.4387264, 3.0539317, 0.4387217, 3.0539322, -2.3511636, 2.3664017

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A1_A2_B1

### Relational analysis result of IS_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129975, upper bound: 1.4142296
time: 5.45 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2

### Relational analysis result of IS_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142299, upper bound: 1.4142295
time: 5.32 seconds

## BFS IS instance: IS_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.6939430, -5.0657420, -9.6733437, -5.0703855, -3.0686836, 2.9875181
1: -15.1012497, -10.8697720, -15.0795126, -10.8686848, -3.1976919, 3.1743994
2: -9.0496235, -5.7592134, -9.0378990, -5.7760863, -2.6813126, 2.7030561
3: -11.5578575, -7.3928409, -11.5115681, -7.4143248, -3.6028161, 3.5673351
4: -5.4727850, -1.8661127, -5.4447594, -1.9613184, -3.1537580, 3.1709390
5: -3.5807779, -0.4715900, -3.5595188, -0.5033293, -2.5560546, 2.5422776
6: -11.5694466, -6.9544353, -11.5687103, -6.9760337, -3.3561468, 3.3380971
7: -2.7788510, 0.8648715, -2.7804103, 0.8226824, -3.2641020, 3.3034577
8: -5.0864167, -1.4727631, -5.0695553, -1.4847116, -2.6887465, 2.7256315
9: 0.3877354, 3.0650821, 0.4508972, 3.0508280, -2.3960352, 2.3593853

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_A1_B1

### Relational analysis result of IS_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4118114, upper bound: 1.4138002
time: 5.09 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2

### Relational analysis result of IS_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4130491, upper bound: 1.4137999
time: 4.62 seconds

## BFS IS instance: IS_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.7308750, -5.0533290, -9.6923943, -5.0648384, -3.0977259, 3.0199096
1: -15.1290913, -10.8281574, -15.0922260, -10.8474159, -3.2493677, 3.2223535
2: -9.0913172, -5.7413602, -9.0594254, -5.7682366, -2.7130518, 2.7482939
3: -11.5770693, -7.3789635, -11.5214596, -7.4080429, -3.6399751, 3.5946150
4: -5.5142632, -1.8564311, -5.4654369, -1.9569877, -3.1969848, 3.2066362
5: -3.6007345, -0.4574785, -3.5687127, -0.4961605, -2.5835223, 2.5582387
6: -11.5934181, -6.9432263, -11.5807743, -6.9714098, -3.3824320, 3.3635454
7: -2.8240209, 0.8757863, -2.8032002, 0.8274655, -3.3206797, 3.3509235
8: -5.0960493, -1.4535069, -5.0741334, -1.4750609, -2.7081118, 2.7465739
9: 0.3626266, 3.0718713, 0.4387217, 3.0539322, -2.4248443, 2.3872097

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4147631
time: 5.23 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142299, upper bound: 1.4147627
time: 5.66 seconds

## BFS IS instance: IS_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -9.6724758, -5.1513062, -9.7262812, -5.0818281, -3.0129128, 2.9805405
1: -15.0560608, -10.9625511, -15.1199627, -10.8536339, -3.1123099, 3.1115131
2: -9.0446472, -5.8387461, -9.0882931, -5.7572312, -2.6943603, 2.6742802
3: -11.5060225, -7.4424539, -11.5749512, -7.3916945, -3.5748777, 3.5872388
4: -5.4292078, -2.0310245, -5.5020175, -1.8715513, -3.1250038, 3.1154954
5: -3.4420493, -0.5120606, -3.5717914, -0.4645195, -2.4069767, 2.4574294
6: -11.4902897, -6.9897213, -11.5733356, -6.9502792, -3.2869139, 3.2695498
7: -2.7814407, 0.8091979, -2.8165984, 0.8715315, -3.3248544, 3.2867074
8: -5.0576382, -1.5581932, -5.0900240, -1.4723272, -2.6060209, 2.6098123
9: 0.4686995, 2.9883897, 0.3694859, 3.0573421, -2.2954035, 2.3441432

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of IS_B2_A2_A1_A1_B1

### Relational analysis result of IS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135294, upper bound: 1.4116666
time: 4.91 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2

### Relational analysis result of IS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135295, upper bound: 1.4116662
time: 5.53 seconds

## BFS IS instance: IS_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -9.7258587, -5.1430144, -9.7267847, -5.0808225, -3.0383964, 2.9961886
1: -15.0851736, -10.9182358, -15.1211395, -10.8533707, -3.1386533, 3.1433575
2: -9.0585203, -5.8056507, -9.0888939, -5.7566819, -2.7141852, 2.7210426
3: -11.5458851, -7.4240246, -11.5753622, -7.3910599, -3.6193094, 3.6029549
4: -5.4661131, -1.9609370, -5.5042033, -1.8710210, -3.1639581, 3.1506307
5: -3.4679306, -0.4743328, -3.5724616, -0.4640818, -2.4289131, 2.4692988
6: -11.5032864, -6.9683542, -11.5739641, -6.9501362, -3.3121958, 3.2833972
7: -2.8342905, 0.8228474, -2.8170152, 0.8726463, -3.3477564, 3.3170967
8: -5.0839467, -1.5327125, -5.0906010, -1.4719934, -2.6253457, 2.6439905
9: 0.4182305, 3.0011253, 0.3687944, 3.0583110, -2.3197250, 2.3580453

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6139

## Relational analysis of IS_B2_A2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135860, upper bound: 1.4119235
time: 4.83 seconds

## Relational analysis of IS_B2_A2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147619, upper bound: 1.4128885
time: 5.22 seconds

## BFS IS instance: IS_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -9.6531181, -5.0749526, -9.7236881, -5.0555067, -3.0660400, 2.9813285
1: -15.0505028, -10.8621016, -15.1205711, -10.8308125, -3.1689243, 3.1418965
2: -9.0435181, -5.8156915, -9.0895929, -5.7502437, -2.7012224, 2.6902635
3: -11.5069675, -7.4414186, -11.5759344, -7.3851085, -3.5807467, 3.5852318
4: -5.4352460, -1.9753574, -5.5069466, -1.8580804, -3.1879678, 3.1524882
5: -3.5155721, -0.5048790, -3.5900948, -0.4593525, -2.4274783, 2.5341125
6: -11.5163689, -6.9829216, -11.5815449, -6.9457197, -3.2872138, 3.3095665
7: -2.7710075, 0.8179655, -2.8176808, 0.8745742, -3.3297262, 3.2974453
8: -5.0558014, -1.5111427, -5.0912294, -1.4603081, -2.6303062, 2.6297972
9: 0.4571071, 3.0395703, 0.3650165, 3.0695546, -2.3531718, 2.3652050

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B2_A2_A2_A1_A1

### Relational analysis result of IS_B2_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147622, upper bound: 1.4131991
time: 6.51 seconds

## Relational analysis of IS_B2_A2_A2_A1_A2

### Relational analysis result of IS_B2_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147617, upper bound: 1.4144235
time: 5.41 seconds

## BFS IS instance: IS_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -9.6942692, -5.0617027, -9.7316399, -5.0531874, -3.0948591, 3.0143735
1: -15.0952206, -10.8431873, -15.1295853, -10.8272858, -3.2347536, 3.2238541
2: -9.0615816, -5.7651782, -9.0930614, -5.7406940, -2.7273674, 2.7339733
3: -11.5230570, -7.4036927, -11.5797920, -7.3782058, -3.6232233, 3.6384416
4: -5.4777107, -1.9554585, -5.5150676, -1.8537846, -3.2288761, 3.1769571
5: -3.5736198, -0.4953156, -3.6012421, -0.4570112, -2.5444460, 2.6160867
6: -11.5837059, -6.9705009, -11.5943956, -6.9428940, -3.3862944, 3.3675442
7: -2.8097568, 0.8291922, -2.8250465, 0.8776369, -3.3649111, 3.3455844
8: -5.0775580, -1.4732771, -5.0968323, -1.4531312, -2.7732773, 2.7532811
9: 0.4356971, 3.0562396, 0.3610530, 3.0728159, -2.3966026, 2.4157553

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B2_A2_A2_A2_A1

### Relational analysis result of IS_B2_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147621, upper bound: 1.4135289
time: 7.78 seconds

## Relational analysis of IS_B2_A2_A2_A2_A2

### Relational analysis result of IS_B2_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147616, upper bound: 1.4147603
time: 5.51 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.22 seconds
IS_B1_A2_A1_A1_A1, status: Status.VERIFIED, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4130486, upper bound: 1.4113864
IS_B1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4130484, upper bound: 1.4132611
IS_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4129975, upper bound: 1.4142296
IS_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4142299, upper bound: 1.4142295
IS_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4118114, upper bound: 1.4138002
IS_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4130491, upper bound: 1.4137999
IS_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4147631
IS_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4142299, upper bound: 1.4147627
IS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4135294, upper bound: 1.4116666
IS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4135295, upper bound: 1.4116662
IS_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4135860, upper bound: 1.4119235
IS_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4147619, upper bound: 1.4128885
IS_B2_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4147622, upper bound: 1.4131991
IS_B2_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4147617, upper bound: 1.4144235
IS_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4147621, upper bound: 1.4135289
IS_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 28.22
Output dim: 9, lower bound: -1.4147616, upper bound: 1.4147603

## BFS IS instance: IS_B1_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -9.6547890, -5.0769014, -9.6733446, -5.0703845, -3.0294518, 2.9404359
1: -15.0645342, -10.8892708, -15.0795116, -10.8686848, -3.1589570, 3.1271522
2: -9.0170021, -5.7857342, -9.0378990, -5.7760863, -2.6437950, 2.6487541
3: -11.5016794, -7.4215121, -11.5115681, -7.4143257, -3.5523853, 3.5531559
4: -5.4243236, -1.9664223, -5.4447579, -1.9613178, -3.1063442, 3.0816383
5: -3.5490358, -0.5102730, -3.5595188, -0.5033283, -2.4944925, 2.5252171
6: -11.5567656, -6.9825134, -11.5687103, -6.9760323, -3.3357863, 3.3263650
7: -2.7580986, 0.8165779, -2.7804089, 0.8226824, -3.2413998, 3.2723937
8: -5.0646868, -1.4944301, -5.0695553, -1.4847121, -2.6867046, 2.7047465
9: 0.4636345, 3.0470600, 0.4508972, 3.0508280, -2.3219614, 2.3186307

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B1_A2_A1_A1_A2_A1

### Relational analysis result of IS_B1_A2_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4130480, upper bound: 1.4129146
time: 4.60 seconds

## Relational analysis of IS_B1_A2_A1_A1_A2_A2

### Relational analysis result of IS_B1_A2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4130479, upper bound: 1.4132606
time: 4.40 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -9.6920061, -5.0658555, -9.6915245, -5.0671000, -3.0567813, 3.0057421
1: -15.0910683, -10.8475685, -15.0896091, -10.8477535, -3.2087393, 3.1988444
2: -9.0589466, -5.7687302, -9.0583553, -5.7693439, -2.6742010, 2.7146921
3: -11.5211916, -7.4086246, -11.5208569, -7.4093404, -3.5873094, 3.5687170
4: -5.4632359, -1.9573025, -5.4604721, -1.9576896, -3.1458778, 3.1378632
5: -3.5680780, -0.4965296, -3.5672817, -0.4970446, -2.5459051, 2.5388865
6: -11.5802650, -6.9715438, -11.5796518, -6.9717097, -3.3674154, 3.3346395
7: -2.8027992, 0.8264308, -2.8023043, 0.8251877, -3.2947044, 3.3005428
8: -5.0736670, -1.4753838, -5.0730290, -1.4757948, -2.6846209, 2.7237430
9: 0.4393468, 3.0529432, 0.4401393, 3.0517116, -2.3483858, 2.3641253

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A1_A2_B1_A1

### Relational analysis result of IS_B1_A2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4129964
time: 4.97 seconds

## Relational analysis of IS_B1_A2_A1_A2_B1_A2

### Relational analysis result of IS_B1_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4142293
time: 5.10 seconds

## BFS IS instance: IS_B1_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -9.6923847, -5.0648580, -9.7449207, -5.0590014, -3.0728416, 3.0535424
1: -15.0922108, -10.8474226, -15.1206589, -10.8046227, -3.2537193, 3.2486830
2: -9.0594177, -5.7682462, -9.0725355, -5.7367778, -2.7283945, 2.7487130
3: -11.5214558, -7.4080510, -11.5599260, -7.3902893, -3.6101785, 3.6122990
4: -5.4654078, -1.9569938, -5.4976149, -1.8877120, -3.1976156, 3.1868048
5: -3.5687029, -0.4961653, -3.5930345, -0.4576855, -2.5665565, 2.5641580
6: -11.5807638, -6.9714127, -11.5926247, -6.9502172, -3.4009643, 3.3642735
7: -2.8031917, 0.8274503, -2.8546848, 0.8388014, -3.3264065, 3.3561854
8: -5.0741243, -1.4750671, -5.1007357, -1.4508238, -2.7175264, 2.7710419
9: 0.4387355, 3.0539179, 0.3890657, 3.0643830, -2.3621507, 2.4122953

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 485
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B1_A2_A1_A2_B2_A1

### Relational analysis result of IS_B1_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142295, upper bound: 1.4138920
time: 5.08 seconds

## Relational analysis of IS_B1_A2_A1_A2_B2_A2

### Relational analysis result of IS_B1_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142294, upper bound: 1.4142288
time: 5.15 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -9.6934004, -5.0668383, -9.6724749, -5.0726466, -3.0659723, 2.9855223
1: -15.0999851, -10.8700686, -15.0769091, -10.8690186, -3.1957693, 3.1707506
2: -9.0490036, -5.7598543, -9.0368233, -5.7771764, -2.6792269, 2.7007964
3: -11.5573645, -7.3935051, -11.5109663, -7.4156923, -3.6004591, 3.5656347
4: -5.4704590, -1.8666694, -5.4397850, -1.9620051, -3.1504526, 3.1651883
5: -3.5800452, -0.4720726, -3.5581019, -0.5042143, -2.5537596, 2.5397294
6: -11.5687838, -6.9545975, -11.5675888, -6.9763346, -3.3544054, 3.3359976
7: -2.7783432, 0.8637137, -2.7795062, 0.8204098, -3.2608833, 3.3011870
8: -5.0857944, -1.4732080, -5.0684528, -1.4854469, -2.6869822, 2.7235551
9: 0.3884668, 3.0640926, 0.4523096, 3.0486073, -2.3931355, 2.3570597

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4118114, upper bound: 1.4125615
time: 4.65 seconds

## Relational analysis of IS_B1_A2_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4118114, upper bound: 1.4137998
time: 4.82 seconds

## BFS IS instance: IS_B1_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -9.6939602, -5.0655365, -9.7262421, -5.0642123, -3.0777445, 3.0239167
1: -15.1014519, -10.8696518, -15.1085606, -10.8250351, -3.2144518, 3.2213807
2: -9.0496368, -5.7589970, -9.0514526, -5.7440267, -2.7268219, 2.7356415
3: -11.5580187, -7.3927898, -11.5506630, -7.3963299, -3.6169958, 3.6096239
4: -5.4730072, -1.8660617, -5.4774389, -1.8917803, -3.1988907, 3.2046216
5: -3.5809216, -0.4715710, -3.5841665, -0.4646778, -2.5702963, 2.5622272
6: -11.5694962, -6.9544191, -11.5808477, -6.9548035, -3.3891578, 3.3629515
7: -2.7790246, 0.8649573, -2.8322780, 0.8340716, -3.2930250, 3.3251603
8: -5.0864458, -1.4725456, -5.0966272, -1.4599981, -2.7203169, 2.7624650
9: 0.3876829, 3.0650878, 0.4010234, 3.0613585, -2.4070683, 2.4056838

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 485

## Relational analysis of IS_B1_A2_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -1.4130478, upper bound: 1.4119236
time: 4.59 seconds

## Relational analysis of IS_B1_A2_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4130476, upper bound: 1.4137983
time: 4.38 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.7303391, -5.0543413, -9.6915245, -5.0671000, -3.0950198, 3.0179963
1: -15.1279087, -10.8284264, -15.0896091, -10.8477535, -3.2475100, 3.2187214
2: -9.0906982, -5.7419224, -9.0583553, -5.7693439, -2.7110109, 2.7461264
3: -11.5766430, -7.3796120, -11.5208569, -7.4093404, -3.6376963, 3.5929432
4: -5.5120449, -1.8569800, -5.4604721, -1.9576896, -3.1938009, 3.2008975
5: -3.6000504, -0.4579496, -3.5672817, -0.4970446, -2.5813017, 2.5556965
6: -11.5927725, -6.9433742, -11.5796518, -6.9717097, -3.3807154, 3.3614793
7: -2.8235941, 0.8746576, -2.8023043, 0.8251877, -3.3175268, 3.3487005
8: -5.0954399, -1.4538546, -5.0730290, -1.4757948, -2.7063560, 2.7445307
9: 0.3633394, 3.0708878, 0.4401393, 3.0517116, -2.4219699, 2.3849256

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 6155
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of IS_B1_A2_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4135294
time: 5.30 seconds

## Relational analysis of IS_B1_A2_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4147629
time: 5.09 seconds

## BFS IS instance: IS_B1_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.7308664, -5.0533400, -9.7453146, -5.0586390, -3.1075320, 3.0650039
1: -15.1290789, -10.8281574, -15.1213217, -10.8038864, -3.2788153, 3.2693195
2: -9.0913105, -5.7413673, -9.0728416, -5.7361302, -2.7579470, 2.7806792
3: -11.5770626, -7.3789701, -11.5604630, -7.3900709, -3.6541047, 3.6363935
4: -5.5142412, -1.8564372, -5.4981346, -1.8873448, -3.2415195, 3.2404454
5: -3.6007259, -0.4574823, -3.5933709, -0.4574642, -2.6008930, 2.5780609
6: -11.5934105, -6.9432282, -11.5931091, -6.9501491, -3.4138703, 3.3874400
7: -2.8240147, 0.8757691, -2.8551264, 0.8389716, -3.3495531, 3.3774338
8: -5.0960441, -1.4535108, -5.1013203, -1.4503083, -2.7397876, 2.7834816
9: 0.3626361, 3.0718608, 0.3887391, 3.0644593, -2.4357929, 2.4326429

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 485
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B1_A2_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142295, upper bound: 1.4144251
time: 5.20 seconds

## Relational analysis of IS_B1_A2_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4142294, upper bound: 1.4147616
time: 5.21 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -9.6724758, -5.1513062, -9.7257929, -5.0830851, -3.0114980, 2.9800630
1: -15.0560608, -10.9625511, -15.1185226, -10.8538265, -3.1121035, 3.1097403
2: -9.0446472, -5.8387461, -9.0877333, -5.7578573, -2.6936350, 2.6733546
3: -11.5060225, -7.4424539, -11.5745983, -7.3924427, -3.5736704, 3.5868425
4: -5.4292078, -2.0310245, -5.4992743, -1.8719554, -3.1245708, 3.1123505
5: -3.4420493, -0.5120606, -3.5709753, -0.4650087, -2.4061966, 2.4565361
6: -11.4902897, -6.9897213, -11.5727291, -6.9504499, -3.2864280, 3.2688682
7: -2.7814407, 0.8091979, -2.8161020, 0.8702917, -3.3231974, 3.2861814
8: -5.0576382, -1.5581932, -5.0894065, -1.4727249, -2.6055818, 2.6088638
9: 0.4686995, 2.9883897, 0.3702712, 3.0561337, -2.2942381, 2.3434141

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B2_A2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135291, upper bound: 1.4113356
time: 5.27 seconds

## Relational analysis of IS_B2_A2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135290, upper bound: 1.4116685
time: 5.71 seconds

## BFS IS instance: IS_B2_A2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -9.6724758, -5.1513062, -9.7778654, -5.0756927, -3.0197883, 3.0094798
1: -15.0560608, -10.9625511, -15.1405582, -10.8111305, -3.1255403, 3.1314235
2: -9.0446472, -5.8387461, -9.1027031, -5.7267356, -2.7113438, 2.6949720
3: -11.5060225, -7.4424539, -11.6138077, -7.3757677, -3.5878997, 3.6060433
4: -5.4292078, -2.0310245, -5.5338259, -1.8025482, -3.1439886, 3.1484628
5: -3.4420493, -0.5120606, -3.5956564, -0.4358473, -2.4134703, 2.4776022
6: -11.4902897, -6.9897213, -11.5840368, -6.9294324, -3.3060837, 3.2815952
7: -2.7814407, 0.8091979, -2.8674288, 0.8821292, -3.3352623, 3.3401065
8: -5.0576382, -1.5581932, -5.1100731, -1.4487739, -2.6187968, 2.6412880
9: 0.4686995, 2.9883897, 0.3239961, 3.0687609, -2.3066180, 2.3636768

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B2_A2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135291, upper bound: 1.4113366
time: 7.48 seconds

## Relational analysis of IS_B2_A2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135290, upper bound: 1.4116657
time: 6.74 seconds

## BFS IS instance: IS_B2_A2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.6883469, -5.1551065, -9.7078981, -5.0864959, -2.9940395, 2.9559033
1: -15.0580320, -10.9603167, -15.1083870, -10.8747759, -3.0748582, 3.0862608
2: -9.0160732, -5.8230267, -9.0671644, -5.7647767, -2.6648393, 2.6679084
3: -11.5263872, -7.4370937, -11.5651217, -7.3976941, -3.5919237, 3.5756807
4: -5.4247150, -1.9704455, -5.4831157, -1.8761008, -3.1162844, 3.1150331
5: -3.4487631, -0.4885035, -3.5630665, -0.4713459, -2.3982468, 2.4462092
6: -11.4788647, -6.9794741, -11.5616188, -6.9548864, -3.2842846, 3.2572088
7: -2.7888722, 0.8118687, -2.7939510, 0.8673615, -3.2907920, 3.2750921
8: -5.0744972, -1.5521894, -5.0856361, -1.4816008, -2.6043706, 2.6204648
9: 0.4433279, 2.9943125, 0.3814955, 3.0549984, -2.2902255, 2.3377335

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6155
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B2_A2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135856, upper bound: 1.4115766
time: 4.27 seconds

## Relational analysis of IS_B2_A2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4135855, upper bound: 1.4119230
time: 4.63 seconds

## BFS IS instance: IS_B2_A2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -9.7258453, -5.1430178, -9.7266502, -5.0808277, -3.0203876, 2.9960661
1: -15.0851650, -10.9182711, -15.1211023, -10.8534241, -3.1372905, 3.1346364
2: -9.0585079, -5.8056583, -9.0888586, -5.7567477, -2.6959238, 2.7189748
3: -11.5458765, -7.4240303, -11.5751982, -7.3910851, -3.6291237, 3.6018138
4: -5.4661040, -1.9609501, -5.5041747, -1.8712497, -3.1585336, 3.1501760
5: -3.4679258, -0.4743423, -3.5724344, -0.4641695, -2.4282541, 2.4607990
6: -11.5032673, -6.9683557, -11.5738726, -6.9501486, -3.3087301, 3.2816815
7: -2.8342853, 0.8228426, -2.8169818, 0.8725848, -3.3475060, 3.3230534
8: -5.0839262, -1.5327158, -5.0905452, -1.4720206, -2.6250672, 2.6426427
9: 0.4182410, 3.0011215, 0.3688927, 3.0582957, -2.3191590, 2.3644040

Time for backsubstitution: 14.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6155
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 6238
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6165

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6155

## Relational analysis of IS_B2_A2_A1_A2_A2_A1

### Relational analysis result of IS_B2_A2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147615, upper bound: 1.4125516
time: 8.93 seconds

## Relational analysis of IS_B2_A2_A1_A2_A2_A2

### Relational analysis result of IS_B2_A2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147614, upper bound: 1.4128885
time: 4.85 seconds

## BFS IS instance: IS_B2_A2_A2_A1_A1

### Backsubstitution after applying IS history:
0: -9.6522751, -5.0772114, -9.7231655, -5.0565186, -3.0642424, 2.9787118
1: -15.0477104, -10.8624401, -15.1193600, -10.8310766, -3.1655469, 3.1400208
2: -9.0424652, -5.8168249, -9.0889835, -5.7508020, -2.6991739, 2.6882014
3: -11.5063620, -7.4427123, -11.5755196, -7.3857489, -3.5790563, 3.5830388
4: -5.4302735, -1.9760109, -5.5047317, -1.8586102, -3.1822820, 3.1493559
5: -3.5141115, -0.5055628, -3.5894125, -0.4598017, -2.4249430, 2.5320923
6: -11.5152168, -6.9832010, -11.5809078, -6.9458652, -3.2852097, 3.3081515
7: -2.7701263, 0.8156905, -2.8172545, 0.8734503, -3.3274851, 3.2944317
8: -5.0547171, -1.5118637, -5.0906315, -1.4606538, -2.6282849, 2.6280015
9: 0.4584150, 3.0373466, 0.3657165, 3.0685725, -2.3510029, 2.3623450

Time for backsubstitution: 14.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 6208
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6139
type: B, layer: 1, pos: 6238
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6238

## Relational analysis of IS_B2_A2_A2_A1_A1_A1

### Relational analysis result of IS_B2_A2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147618, upper bound: 1.4126131
time: 5.38 seconds

## Relational analysis of IS_B2_A2_A2_A1_A1_A2

### Relational analysis result of IS_B2_A2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147618, upper bound: 1.4132019
time: 9.25 seconds

## BFS IS instance: IS_B2_A2_A2_A1_A2

### Backsubstitution after applying IS history:
0: -9.7053413, -5.0689435, -9.7236786, -5.0555215, -3.0985136, 2.9886103
1: -15.0749331, -10.8181677, -15.1205568, -10.8308163, -3.2103310, 3.1719751
2: -9.0569515, -5.7838645, -9.0895882, -5.7502508, -2.7323780, 2.7338214
3: -11.5455971, -7.4240870, -11.5759315, -7.3851166, -3.6227388, 3.5987287
4: -5.4672775, -1.9052590, -5.5069242, -1.8580860, -3.2212348, 3.1784086
5: -3.5397265, -0.4702578, -3.5900884, -0.4593573, -2.4467130, 2.5434246
6: -11.5276709, -6.9618897, -11.5815392, -6.9457221, -3.3100510, 3.3219306
7: -2.8225167, 0.8290887, -2.8176737, 0.8745575, -3.3505106, 3.3230629
8: -5.0795422, -1.4866533, -5.0912232, -1.4603138, -2.6611319, 2.6608016
9: 0.4089603, 3.0500357, 0.3650260, 3.0695443, -2.3758264, 2.3762362

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6238
type: A, layer: 1, pos: 6139
type: A, layer: 1, pos: 471
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 6208
type: A, layer: 1, pos: 163
type: B, layer: 1, pos: 163
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 6208
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 6238
type: B, layer: 1, pos: 6139
type: A, layer: 1, pos: 6126
type: B, layer: 1, pos: 6155
type: B, layer: 1, pos: 6165
type: B, layer: 1, pos: 485

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6238

## Relational analysis of IS_B2_A2_A2_A1_A2_A1

### Relational analysis result of IS_B2_A2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147612, upper bound: 1.4138410
time: 7.24 seconds

## Relational analysis of IS_B2_A2_A2_A1_A2_A2

### Relational analysis result of IS_B2_A2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.4147613, upper bound: 1.4144231
time: 5.25 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 27.44 seconds
IS_B1_A2_A1_A1_A2_A1, status: Status.VERIFIED, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4130480, upper bound: 1.4129146
IS_B1_A2_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4130479, upper bound: 1.4132606
IS_B1_A2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4129964
IS_B1_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4142293
IS_B1_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4142295, upper bound: 1.4138920
IS_B1_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4142294, upper bound: 1.4142288
IS_B1_A2_A2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4118114, upper bound: 1.4125615
IS_B1_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4118114, upper bound: 1.4137998
IS_B1_A2_A2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4130478, upper bound: 1.4119236
IS_B1_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4130476, upper bound: 1.4137983
IS_B1_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4135294
IS_B1_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4129974, upper bound: 1.4147629
IS_B1_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4142295, upper bound: 1.4144251
IS_B1_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4142294, upper bound: 1.4147616
IS_B2_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4135291, upper bound: 1.4113356
IS_B2_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4135290, upper bound: 1.4116685
IS_B2_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4135291, upper bound: 1.4113366
IS_B2_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4135290, upper bound: 1.4116657
IS_B2_A2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4135856, upper bound: 1.4115766
IS_B2_A2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4135855, upper bound: 1.4119230
IS_B2_A2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4147615, upper bound: 1.4125516
IS_B2_A2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4147614, upper bound: 1.4128885
IS_B2_A2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4147618, upper bound: 1.4126131
IS_B2_A2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4147618, upper bound: 1.4132019
IS_B2_A2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4147612, upper bound: 1.4138410
IS_B2_A2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 27.44
Output dim: 9, lower bound: -1.4147613, upper bound: 1.4144231
IS_B2_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 27.44
Output dim: 9, lower bound: -1.4147621, upper bound: 1.4135289
IS_B2_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 27.44
Output dim: 9, lower bound: -1.4147616, upper bound: 1.4147603
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.3590686321258545
rel_dist={9: [-1.414766908177059, 1.4147664541724545]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 2420.38 seconds
