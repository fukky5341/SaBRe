## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.719649471


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.0727234, -6.1738672, -8.0727234, -6.1738672, -1.4588056, 1.4588056)
1: (-10.4562082, -8.2853909, -10.4562082, -8.2853909, -1.6418862, 1.6418862)
2: (-4.7416358, -2.8030829, -4.7416358, -2.8030829, -1.3488832, 1.3488827)
3: (-5.6578608, -3.3550725, -5.6578608, -3.3550725, -1.7874470, 1.7874465)
4: (-13.0044861, -10.3705025, -13.0044861, -10.3705025, -1.5689108, 1.5689108)
5: (-3.3171821, -1.8086381, -3.3171821, -1.8086381, -0.9303412, 0.9303412)
6: (-10.5895643, -8.5086870, -10.5895643, -8.5086870, -1.3711376, 1.3711374)
7: (-9.0877266, -6.7382479, -9.0877266, -6.7382479, -2.0479746, 2.0479746)
8: (9.8031464, 11.6969671, 9.8031464, 11.6969671, -1.5339589, 1.5339584)
9: (-7.3276410, -4.8431973, -7.3276410, -4.8431973, -1.8485889, 1.8485889)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.63 + 37.84 = 61.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.7232643, upper bound: 0.7232643

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6127
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 6109
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830
type: A, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6127

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206914, upper bound: 0.7232490
time: 5.70 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232597, upper bound: 0.7232602
time: 6.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.18 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.18
Output dim: 8, lower bound: -0.7206914, upper bound: 0.7232490
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.18
Output dim: 8, lower bound: -0.7232597, upper bound: 0.7232602

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.0498524, -6.1755099, -8.0619402, -6.1746368, -1.4333949, 1.4445422
1: -10.4529991, -8.2873192, -10.4547043, -8.2862740, -1.6247878, 1.6237564
2: -4.7245579, -2.8045940, -4.7335825, -2.8037901, -1.3306198, 1.3396058
3: -5.6533866, -3.3581309, -5.6557779, -3.3565199, -1.7625422, 1.7625027
4: -13.0033913, -10.3824625, -13.0039730, -10.3761501, -1.5639300, 1.5569038
5: -3.3168857, -1.8334845, -3.3170266, -1.8203409, -0.9178455, 0.9046702
6: -10.5887737, -8.5439444, -10.5891953, -8.5253201, -1.3516145, 1.3325706
7: -9.0717640, -6.7400117, -9.0801954, -6.7390790, -2.0271654, 2.0348215
8: 9.8145256, 11.6965656, 9.8085155, 11.6967812, -1.5180631, 1.5233474
9: -7.3242903, -4.8463607, -7.3260803, -4.8446774, -1.8285279, 1.8292141

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204745, upper bound: 0.7203252
time: 6.97 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206855, upper bound: 0.7232428
time: 6.34 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.0787544, -6.1395850, -8.0727148, -6.1738696, -1.4551311, 1.4749665
1: -10.4661427, -8.2740974, -10.4562054, -8.2853928, -1.6427097, 1.6621537
2: -4.7445574, -2.7718077, -4.7416277, -2.8030825, -1.3443022, 1.3737986
3: -5.6787882, -3.3528078, -5.6578579, -3.3550749, -1.8148317, 1.7822104
4: -13.0223980, -10.3628731, -13.0044851, -10.3705149, -1.5844147, 1.5744584
5: -3.3503807, -1.8067336, -3.3171821, -1.8086568, -0.9439838, 0.9193957
6: -10.6416788, -8.4929924, -10.5895634, -8.5087070, -1.3965278, 1.3773835
7: -9.1057091, -6.7119346, -9.0877151, -6.7382498, -2.0628300, 2.0716672
8: 9.7982502, 11.7096462, 9.8031540, 11.6969681, -1.5361242, 1.5472727
9: -7.3555994, -4.8390579, -7.3276386, -4.8431988, -1.8751235, 1.8468099

Time for backsubstitution: 21.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6109
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6109

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230449, upper bound: 0.7203382
time: 6.10 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232538, upper bound: 0.7232553
time: 4.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.77 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 31.77
Output dim: 8, lower bound: -0.7204745, upper bound: 0.7203252
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 31.77
Output dim: 8, lower bound: -0.7206855, upper bound: 0.7232428
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 31.77
Output dim: 8, lower bound: -0.7230449, upper bound: 0.7203382
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 31.77
Output dim: 8, lower bound: -0.7232538, upper bound: 0.7232553

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -8.0485802, -6.1808658, -8.0613823, -6.1770077, -1.4294438, 1.4387805
1: -10.4519701, -8.2913761, -10.4542484, -8.2880802, -1.6220937, 1.6194010
2: -4.7236915, -2.8084431, -4.7332063, -2.8054967, -1.3281999, 1.3352728
3: -5.6410618, -3.3588247, -5.6503029, -3.3568258, -1.7495213, 1.7560706
4: -12.9828806, -10.3850346, -12.9948931, -10.3772755, -1.5421023, 1.5451441
5: -3.3166955, -1.8398135, -3.3169434, -1.8231413, -0.9148705, 0.8983278
6: -10.5885468, -8.5565968, -10.5890942, -8.5309010, -1.3454037, 1.3190036
7: -9.0427456, -6.7400966, -9.0673542, -6.7391176, -1.9973979, 2.0215435
8: 9.8158817, 11.6907244, 9.8091154, 11.6941853, -1.5123520, 1.5148916
9: -7.3141289, -4.8474064, -7.3215594, -4.8451381, -1.8179049, 1.8235068

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204719, upper bound: 0.7186522
time: 5.12 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204720, upper bound: 0.7203227
time: 6.08 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -8.0578833, -6.1748371, -8.0619392, -6.1746392, -1.4418268, 1.4437389
1: -10.4578371, -8.2852135, -10.4547043, -8.2862797, -1.6295681, 1.6254401
2: -4.7337017, -2.8031578, -4.7335820, -2.8037953, -1.3401752, 1.3393726
3: -5.6575961, -3.3442781, -5.6557665, -3.3565226, -1.7625933, 1.7761922
4: -13.0041828, -10.3497639, -13.0039482, -10.3761520, -1.5555878, 1.5728710
5: -3.3268533, -1.8331392, -3.3170259, -1.8203421, -0.9277711, 0.9021297
6: -10.5992632, -8.5404730, -10.5891943, -8.5253286, -1.3624363, 1.3327031
7: -9.0777483, -6.7027478, -9.0801849, -6.7390800, -2.0229759, 2.0552859
8: 9.8025341, 11.6976728, 9.8085155, 11.6967707, -1.5286622, 1.5235591
9: -7.3273993, -4.8373170, -7.3260736, -4.8446803, -1.8293438, 1.8379574

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206830, upper bound: 0.7215684
time: 6.14 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206830, upper bound: 0.7232401
time: 6.77 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -8.0775795, -6.1449413, -8.0721607, -6.1762414, -1.4512525, 1.4692216
1: -10.4651108, -8.2782469, -10.4557514, -8.2872009, -1.6400027, 1.6577764
2: -4.7436996, -2.7756748, -4.7412491, -2.8047915, -1.3418784, 1.3694644
3: -5.6662722, -3.3535008, -5.6523757, -3.3553798, -1.8016415, 1.7757831
4: -13.0018845, -10.3653011, -12.9954052, -10.3716335, -1.5625386, 1.5626585
5: -3.3501916, -1.8130664, -3.3170960, -1.8114557, -0.9402955, 0.9130425
6: -10.6414557, -8.5055237, -10.5894642, -8.5142679, -1.3891931, 1.3633165
7: -9.0767689, -6.7120190, -9.0748940, -6.7382851, -2.0331631, 2.0557256
8: 9.7995939, 11.7038050, 9.8037529, 11.6943741, -1.5304174, 1.5388169
9: -7.3451796, -4.8400860, -7.3231087, -4.8436518, -1.8643508, 1.8411040

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4671

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7213654, upper bound: 0.7203360
time: 6.90 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230423, upper bound: 0.7203374
time: 5.02 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -8.0867958, -6.1389189, -8.0727139, -6.1738720, -1.4634829, 1.4741054
1: -10.4709988, -8.2718964, -10.4562054, -8.2853956, -1.6475105, 1.6637707
2: -4.7536950, -2.7703409, -4.7416267, -2.8030901, -1.3538504, 1.3735833
3: -5.6830764, -3.3389628, -5.6578484, -3.3550761, -1.8149276, 1.7959304
4: -13.0231867, -10.3302145, -13.0044603, -10.3705158, -1.5761645, 1.5866506
5: -3.3603458, -1.8063849, -3.3171806, -1.8086584, -0.9442317, 0.9168587
6: -10.6521673, -8.4892979, -10.5895653, -8.5087194, -1.3972192, 1.3774269
7: -9.1118584, -6.6746693, -9.0877028, -6.7382488, -2.0588074, 2.0723193
8: 9.7862625, 11.7107582, 9.8031540, 11.6969566, -1.5467205, 1.5474825
9: -7.3587761, -4.8299704, -7.3276310, -4.8431988, -1.8760643, 1.8556118

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4671
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 4671

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232512, upper bound: 0.7215779
time: 6.71 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232512, upper bound: 0.7232526
time: 4.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.69 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7204719, upper bound: 0.7186522
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7204720, upper bound: 0.7203227
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7206830, upper bound: 0.7215684
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7206830, upper bound: 0.7232401
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7213654, upper bound: 0.7203360
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7230423, upper bound: 0.7203374
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7232512, upper bound: 0.7215779
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.69
Output dim: 8, lower bound: -0.7232512, upper bound: 0.7232526

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -8.0480728, -6.1859879, -8.0602779, -6.1880207, -1.4180532, 1.4328785
1: -10.4513626, -8.3007460, -10.4529209, -8.3082581, -1.6013393, 1.6088095
2: -4.7233763, -2.8185024, -4.7325325, -2.8271184, -1.3059640, 1.3245115
3: -5.6356616, -3.3593168, -5.6386766, -3.3578863, -1.7430706, 1.7439723
4: -12.9821835, -10.3880730, -12.9934025, -10.3837891, -1.5342054, 1.5394561
5: -3.3134863, -1.8400244, -3.3100247, -1.8235919, -0.9110107, 0.8910022
6: -10.5884142, -8.5603294, -10.5888004, -8.5389509, -1.3347149, 1.3130848
7: -9.0413179, -6.7420454, -9.0642958, -6.7432938, -1.9913492, 2.0154424
8: 9.8181467, 11.6905346, 9.8139858, 11.6937761, -1.5079718, 1.5080090
9: -7.3131762, -4.8479428, -7.3195086, -4.8462906, -1.8155017, 1.8205643

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.7184165, upper bound: 0.7183867
time: 6.74 seconds

## Relational analysis of NS_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204700, upper bound: 0.7186503
time: 4.73 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -8.0485802, -6.1808758, -8.0790119, -6.1766319, -1.4236302, 1.4532385
1: -10.4519672, -8.2913904, -10.4829779, -8.2866411, -1.6131372, 1.6444435
2: -4.7236910, -2.8084559, -4.7610540, -2.8046098, -1.3180046, 1.3532963
3: -5.6410542, -3.3588254, -5.6521964, -3.3412557, -1.7649856, 1.7523665
4: -12.9828806, -10.3850393, -13.0022182, -10.3758516, -1.5403900, 1.5513220
5: -3.3166938, -1.8398137, -3.3170919, -1.8134155, -0.9244576, 0.8949628
6: -10.5885429, -8.5566025, -10.6009254, -8.5272694, -1.3478651, 1.3298838
7: -9.0427427, -6.7400975, -9.0741234, -6.7382231, -1.9956713, 2.0272479
8: 9.8158855, 11.6907234, 9.8076286, 11.7007437, -1.5179291, 1.5150223
9: -7.3141284, -4.8474073, -7.3272958, -4.8448715, -1.8181505, 1.8290586

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5832

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7202083, upper bound: 0.7182653
time: 7.11 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204702, upper bound: 0.7203208
time: 6.06 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -8.0573692, -6.1799593, -8.0608273, -6.1856523, -1.4304271, 1.4378304
1: -10.4572315, -8.2945824, -10.4533768, -8.3064566, -1.6088171, 1.6148491
2: -4.7333889, -2.8132148, -4.7329106, -2.8254189, -1.3179407, 1.3286104
3: -5.6521935, -3.3447652, -5.6441431, -3.3575795, -1.7561426, 1.7641025
4: -13.0034857, -10.3528013, -13.0024576, -10.3826666, -1.5476904, 1.5664595
5: -3.3236423, -1.8333484, -3.3101068, -1.8207937, -0.9233689, 0.8948028
6: -10.5991325, -8.5442009, -10.5889006, -8.5333710, -1.3517466, 1.3267853
7: -9.0763092, -6.7046962, -9.0771198, -6.7432542, -2.0169005, 2.0486882
8: 9.8048000, 11.6974850, 9.8133850, 11.6963634, -1.5242829, 1.5166755
9: -7.3264508, -4.8378510, -7.3240185, -4.8458328, -1.8269424, 1.8350134

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7186293, upper bound: 0.7213045
time: 7.27 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206811, upper bound: 0.7215678
time: 6.14 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -8.0578823, -6.1748443, -8.0795689, -6.1742668, -1.4360118, 1.4581008
1: -10.4578362, -8.2852268, -10.4834309, -8.2848396, -1.6206059, 1.6503348
2: -4.7337003, -2.8031702, -4.7614317, -2.8029084, -1.3299799, 1.3574133
3: -5.6575875, -3.3442793, -5.6576614, -3.3409517, -1.7780600, 1.7724977
4: -13.0041800, -10.3497677, -13.0112715, -10.3747330, -1.5538707, 1.5727491
5: -3.3268499, -1.8331387, -3.3171747, -1.8106170, -0.9283936, 0.8987646
6: -10.5992622, -8.5404787, -10.6010246, -8.5217009, -1.3649073, 1.3435848
7: -9.0777454, -6.7027493, -9.0869465, -6.7381835, -2.0212469, 2.0569263
8: 9.8025379, 11.6976738, 9.8070307, 11.7033281, -1.5335236, 1.5236926
9: -7.3274002, -4.8373179, -7.3318224, -4.8444138, -1.8295889, 1.8435249

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5832
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 6127
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5832

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204193, upper bound: 0.7211847
time: 5.94 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7206813, upper bound: 0.7232396
time: 7.87 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -8.0764341, -6.1559582, -8.0716534, -6.1813612, -1.4453082, 1.4577959
1: -10.4637585, -8.2984076, -10.4551392, -8.2965736, -1.6293716, 1.6370435
2: -4.7430382, -2.7972760, -4.7409368, -2.8148670, -1.3311067, 1.3472271
3: -5.6546783, -3.3545542, -5.6469746, -3.3558698, -1.7895794, 1.7692986
4: -13.0003910, -10.3718081, -12.9947109, -10.3746729, -1.5561347, 1.5547645
5: -3.3432770, -1.8135146, -3.3138878, -1.8116643, -0.9329112, 0.9091725
6: -10.6411619, -8.5135670, -10.5893316, -8.5180178, -1.3821635, 1.3527129
7: -9.0737343, -6.7162008, -9.0734749, -6.7402372, -2.0270796, 2.0495791
8: 9.8044643, 11.7033978, 9.8060188, 11.6941833, -1.5235291, 1.5344362
9: -7.3431158, -4.8412371, -7.3221636, -4.8441887, -1.8613214, 1.8387117

Time for backsubstitution: 22.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7193069, upper bound: 0.7200720
time: 6.51 seconds

## Relational analysis of NS_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7213633, upper bound: 0.7203356
time: 6.97 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -8.0951958, -6.1445694, -8.0721617, -6.1762476, -1.4670129, 1.4634252
1: -10.4938259, -8.2767944, -10.4557495, -8.2872143, -1.6605148, 1.6488371
2: -4.7715454, -2.7747664, -4.7412486, -2.8048038, -1.3594291, 1.3592737
3: -5.6681957, -3.3379364, -5.6523685, -3.3553808, -1.7979746, 1.7912493
4: -13.0092039, -10.3639145, -12.9954042, -10.3716364, -1.5624235, 1.5609658
5: -3.3503425, -1.8033426, -3.3170941, -1.8114569, -0.9366057, 0.9227059
6: -10.6532812, -8.5021229, -10.5894651, -8.5142727, -1.3889358, 1.3661985
7: -9.0834837, -6.7111263, -9.0748920, -6.7382870, -2.0387917, 2.0540087
8: 9.7981176, 11.7103634, 9.8037567, 11.6943703, -1.5305424, 1.5416420
9: -7.3510346, -4.8398099, -7.3231058, -4.8436537, -1.8674288, 1.8413701

Time for backsubstitution: 23.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A2_A1_A2_A1

### Relational analysis result of NS_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7209817, upper bound: 0.7200737
time: 4.72 seconds

## Relational analysis of NS_A2_A1_A2_A2

### Relational analysis result of NS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7230403, upper bound: 0.7203357
time: 5.05 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -8.0862617, -6.1440430, -8.0716028, -6.1848850, -1.4520588, 1.4667709
1: -10.4703798, -8.2812576, -10.4548683, -8.3055763, -1.6267395, 1.6531901
2: -4.7533913, -2.7803936, -4.7409558, -2.8247166, -1.3316083, 1.3601832
3: -5.6776929, -3.3394465, -5.6462240, -3.3561347, -1.8084669, 1.7838268
4: -13.0224895, -10.3332481, -13.0029697, -10.3770323, -1.5682178, 1.5802464
5: -3.3571377, -1.8065933, -3.3102639, -1.8091084, -0.9394135, 0.9095254
6: -10.6520357, -8.4930315, -10.5892715, -8.5167837, -1.3864977, 1.3715508
7: -9.1104336, -6.6766233, -9.0846472, -6.7424273, -2.0527401, 2.0657229
8: 9.7885237, 11.7105684, 9.8080273, 11.6965504, -1.5423365, 1.5406003
9: -7.3578234, -4.8305054, -7.3255873, -4.8443551, -1.8736410, 1.8526826

Time for backsubstitution: 23.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211906, upper bound: 0.7213126
time: 6.72 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232493, upper bound: 0.7215775
time: 6.98 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -8.0867920, -6.1389265, -8.0903187, -6.1734991, -1.4576693, 1.4760451
1: -10.4709949, -8.2719088, -10.4849319, -8.2839432, -1.6385608, 1.6691623
2: -4.7536960, -2.7703538, -4.7694745, -2.8022022, -1.3436565, 1.3741083
3: -5.6830702, -3.3389647, -5.6597691, -3.3395085, -1.8245411, 1.7922373
4: -13.0231895, -10.3302174, -13.0117817, -10.3691006, -1.5744991, 1.5865312
5: -3.3603437, -1.8063855, -3.3173318, -1.7989309, -0.9444399, 0.9134920
6: -10.6521664, -8.4893064, -10.6013975, -8.5051126, -1.3968062, 1.3883169
7: -9.1118565, -6.6746707, -9.0944500, -6.7373581, -2.0570784, 2.0739386
8: 9.7862644, 11.7107582, 9.8016729, 11.7035160, -1.5520291, 1.5464330
9: -7.3587732, -4.8299713, -7.3333950, -4.8429322, -1.8763080, 1.8612218

Time for backsubstitution: 23.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5832
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6124
type: A, layer: 1, pos: 6124
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: B, layer: 1, pos: 6109
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: A, layer: 1, pos: 4671
type: B, layer: 1, pos: 6137
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5832

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7211905, upper bound: 0.7229873
time: 4.18 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7232493, upper bound: 0.7232509
time: 4.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.38 seconds
NS_A1_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7184165, upper bound: 0.7183867
NS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7204700, upper bound: 0.7186503
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7202083, upper bound: 0.7182653
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7204702, upper bound: 0.7203208
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7186293, upper bound: 0.7213045
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7206811, upper bound: 0.7215678
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7204193, upper bound: 0.7211847
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7206813, upper bound: 0.7232396
NS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7193069, upper bound: 0.7200720
NS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7213633, upper bound: 0.7203356
NS_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7209817, upper bound: 0.7200737
NS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7230403, upper bound: 0.7203357
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7211906, upper bound: 0.7213126
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7232493, upper bound: 0.7215775
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7211905, upper bound: 0.7229873
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 32.38
Output dim: 8, lower bound: -0.7232493, upper bound: 0.7232509

## BFS NS instance: NS_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.0495644, -6.1844234, -8.0602789, -6.1880226, -1.4197536, 1.4339480
1: -10.4646730, -8.2998295, -10.4529181, -8.3082657, -1.6146355, 1.6061401
2: -4.7237711, -2.8138494, -4.7325263, -2.8271184, -1.3053894, 1.3295760
3: -5.6373305, -3.3560340, -5.6386733, -3.3578866, -1.7439365, 1.7470903
4: -12.9831409, -10.3774223, -12.9933987, -10.3837929, -1.5321970, 1.5498943
5: -3.3142512, -1.8395106, -3.3100221, -1.8235927, -0.9127083, 0.8903565
6: -10.5944147, -8.5592527, -10.5887985, -8.5389557, -1.3405204, 1.3115344
7: -9.0456944, -6.7401142, -9.0642929, -6.7432947, -1.9963102, 2.0163221
8: 9.8162308, 11.6957655, 9.8139896, 11.6937761, -1.5075645, 1.5134215
9: -7.3185959, -4.8469248, -7.3195076, -4.8462925, -1.8184934, 1.8200588

Time for backsubstitution: 23.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6127
type: A, layer: 1, pos: 6124
type: B, layer: 1, pos: 6124
type: B, layer: 1, pos: 6109
type: A, layer: 1, pos: 4556
type: B, layer: 1, pos: 4556
type: A, layer: 1, pos: 4671
type: A, layer: 1, pos: 822
type: B, layer: 1, pos: 822
type: A, layer: 1, pos: 6137
type: B, layer: 1, pos: 5832
type: B, layer: 1, pos: 6137
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 5830
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 5830

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6127

## Relational analysis of NS_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204700, upper bound: 0.7160930
time: 5.97 seconds

## Relational analysis of NS_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.7204701, upper bound: 0.7186503
time: 4.93 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -8.0481510, -6.1819534, -8.0782928, -6.1788673, -1.4198575, 1.4478595
1: -10.4509296, -8.2971401, -10.4809132, -8.2988205, -1.5997491, 1.5888166
2: -4.7215548, -2.8092303, -4.7565317, -2.8062129, -1.3295493, 1.3477197
3: -5.6396165, -3.3596485, -5.6492500, -3.3426957, -1.7609510, 1.7976379
4: -12.9794788, -10.3864946, -12.9950504, -10.3785362, -1.5248218, 1.5420434
5: -3.3160639, -1.8405519, -3.3160241, -1.8149598, -0.9205658, 0.8676932
6: -10.5879459, -8.5593624, -10.5998478, -8.5330744, -1.3411589, 1.3145161
7: -9.0416451, -6.7420158, -9.0719357, -6.7417135, -2.0468884, 2.0228705
8: 9.8191395, 11.6899948, 9.8144674, 11.6996422, -1.5132775, 1.5075965
9: -7.3136063, -4.8497114, -7.3264551, -4.8497591, -1.8100319, 1.8232150

Time for backsubstitution: 22.64 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 61.47 + 550.53 = 612.00 seconds
