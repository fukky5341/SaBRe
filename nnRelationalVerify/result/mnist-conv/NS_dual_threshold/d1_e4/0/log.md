## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.234530954


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5469496, 0.5469496)
1: (-9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4642291, 0.4642291)
2: (-0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4811931, 0.4811933)
3: (4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6157951, 0.6157951)
4: (-10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4521086, 0.4521086)
5: (-4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.3000343, 0.3000343)
6: (-9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3828177, 0.3828173)
7: (-5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6098495, 0.6098495)
8: (-2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4605663, 0.4605663)
9: (-6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4353075, 0.4353075)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.26 + 36.69 = 58.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2393172, upper bound: 0.2393173

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 511

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387026
time: 4.12 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2393161
time: 5.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.89 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.89
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387026
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.89
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2393161

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.7005424, -8.8567333, -9.7107344, -8.8438673, -0.5258007, 0.5168493
1: -9.3144932, -8.5820627, -9.3210011, -8.5770912, -0.4482164, 0.4496257
2: -0.2892741, 0.3964887, -0.2953893, 0.4009780, -0.4624929, 0.4619558
3: 4.1511898, 4.9600511, 4.1439342, 4.9620333, -0.5922284, 0.5950923
4: -10.6752739, -9.8232450, -10.6870317, -9.8161221, -0.4299445, 0.4342468
5: -4.2518725, -3.6431301, -4.2561474, -3.6389291, -0.2878036, 0.2878523
6: -9.4135218, -8.5854979, -9.4167814, -8.5793028, -0.3722217, 0.3703349
7: -5.5471950, -4.7415795, -5.5562115, -4.7323771, -0.5886807, 0.5837073
8: -2.0102525, -1.2585731, -2.0242577, -1.2505159, -0.4345679, 0.4401586
9: -6.0299177, -5.4221191, -6.0373154, -5.4088159, -0.4140873, 0.4088018

Time for backsubstitution: 21.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 511

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 106

### Candidate
type: B, layer: 1, pos: 511

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387022
time: 6.33 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387023
time: 3.87 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.7220135, -8.8412838, -9.7220173, -8.8412819, -0.5452893, 0.5496175
1: -9.3284950, -8.5760498, -9.3284979, -8.5760479, -0.4622154, 0.4642248
2: -0.3016455, 0.4025380, -0.3016478, 0.4025379, -0.4748538, 0.4780598
3: 4.1410475, 4.9639044, 4.1410475, 4.9639049, -0.6126671, 0.6136622
4: -10.6877155, -9.8063335, -10.6877155, -9.8063297, -0.4521041, 0.4458306
5: -4.2563906, -3.6340444, -4.2563906, -3.6340442, -0.2980864, 0.2920208
6: -9.4213018, -8.5785809, -9.4213047, -8.5785809, -0.3743167, 0.3828146
7: -5.5672989, -4.7302260, -5.5673022, -4.7302256, -0.6084323, 0.6141286
8: -2.0258436, -1.2397356, -2.0258422, -1.2397346, -0.4582586, 0.4511800
9: -6.0475612, -5.4065604, -6.0475645, -5.4065604, -0.4313564, 0.4323795

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 511

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 106

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2382617
time: 3.21 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2375027
time: 3.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.71 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387022
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387023
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2382617
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2375027

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -9.7005424, -8.8567333, -9.7005424, -8.8567333, -0.5105333, 0.5105333
1: -9.3144932, -8.5820627, -9.3144932, -8.5820627, -0.4429150, 0.4429150
2: -0.2892741, 0.3964887, -0.2892741, 0.3964887, -0.4546664, 0.4546664
3: 4.1511898, 4.9600511, 4.1511898, 4.9600511, -0.5857549, 0.5857546
4: -10.6752739, -9.8232450, -10.6752739, -9.8232450, -0.4223998, 0.4223998
5: -4.2518725, -3.6431301, -4.2518725, -3.6431301, -0.2825739, 0.2825739
6: -9.4135218, -8.5854979, -9.4135218, -8.5854979, -0.3662522, 0.3662519
7: -5.5471950, -4.7415795, -5.5471950, -4.7415795, -0.5770540, 0.5770540
8: -2.0102525, -1.2585731, -2.0102525, -1.2585731, -0.4265399, 0.4265399
9: -6.0299177, -5.4221191, -6.0299177, -5.4221191, -0.4008839, 0.4008839

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 106

### Candidate
type: A, layer: 1, pos: 106

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -9.7005424, -8.8567333, -9.7219696, -8.8417940, -0.5232213, 0.5258753
1: -9.3144932, -8.5820627, -9.3284864, -8.5762644, -0.4493260, 0.4574988
2: -0.2892741, 0.3964887, -0.3015026, 0.4025243, -0.4620645, 0.4701343
3: 4.1511898, 4.9600511, 4.1411657, 4.9639053, -0.5998034, 0.5970860
4: -10.6752739, -9.8232450, -10.6876116, -9.8063412, -0.4362993, 0.4346561
5: -4.2518725, -3.6431301, -4.2563901, -3.6341074, -0.2905818, 0.2866755
6: -9.4135218, -8.5854979, -9.4212999, -8.5786591, -0.3729916, 0.3749638
7: -5.5471950, -4.7415795, -5.5672779, -4.7306304, -0.5868492, 0.5973821
8: -2.0102525, -1.2585731, -2.0255909, -1.2397461, -0.4382081, 0.4391894
9: -6.0299177, -5.4221191, -6.0475607, -5.4069166, -0.4127088, 0.4107640

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 106

### Candidate
type: B, layer: 1, pos: 106

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -9.7220135, -8.8412838, -9.7220192, -8.8412800, -0.5452890, 0.5496175
1: -9.3284950, -8.5760498, -9.3284969, -8.5760469, -0.4622157, 0.4642243
2: -0.3016455, 0.4025380, -0.3016474, 0.4025371, -0.4748533, 0.4780595
3: 4.1410475, 4.9639044, 4.1410475, 4.9639039, -0.6126676, 0.6136627
4: -10.6877155, -9.8063335, -10.6877155, -9.8063307, -0.4521043, 0.4458303
5: -4.2563906, -3.6340444, -4.2563910, -3.6340427, -0.2980864, 0.2920206
6: -9.4213018, -8.5785809, -9.4213047, -8.5785809, -0.3743169, 0.3828146
7: -5.5672989, -4.7302260, -5.5673027, -4.7302265, -0.6084328, 0.6141286
8: -2.0258436, -1.2397356, -2.0258436, -1.2397342, -0.4582589, 0.4511802
9: -6.0475612, -5.4065604, -6.0475655, -5.4065599, -0.4313567, 0.4323800

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 511

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2373961
time: 3.42 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2375027
time: 3.33 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.7220135, -8.8412838, -9.7220173, -8.8412838, -0.5452893, 0.5496180
1: -9.3284960, -8.5760489, -9.3284988, -8.5760517, -0.4622128, 0.4642255
2: -0.3016450, 0.4025370, -0.3016486, 0.4025355, -0.4748535, 0.4780593
3: 4.1410475, 4.9639049, 4.1410451, 4.9639053, -0.6126671, 0.6136637
4: -10.6877146, -9.8063335, -10.6877203, -9.8063316, -0.4521048, 0.4458351
5: -4.2563910, -3.6340446, -4.2563915, -3.6340427, -0.2980865, 0.2920208
6: -9.4213018, -8.5785809, -9.4213037, -8.5785809, -0.3743160, 0.3828149
7: -5.5673003, -4.7302270, -5.5673022, -4.7302275, -0.6084323, 0.6141281
8: -2.0258427, -1.2397361, -2.0258455, -1.2397337, -0.4582591, 0.4511824
9: -6.0475612, -5.4065619, -6.0475659, -5.4065628, -0.4313552, 0.4323800

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 511

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2373966
time: 3.39 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2375027
time: 3.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.87 seconds
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.87
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2373961
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.87
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2375027
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.87
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2373966
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.87
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2375027

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -9.7220116, -8.8412819, -9.7220192, -8.8412800, -0.5452886, 0.5496182
1: -9.3284960, -8.5760489, -9.3284969, -8.5760469, -0.4622157, 0.4642246
2: -0.3016449, 0.4025385, -0.3016474, 0.4025371, -0.4748528, 0.4780598
3: 4.1410470, 4.9639034, 4.1410475, 4.9639039, -0.6126676, 0.6136632
4: -10.6877155, -9.8063335, -10.6877155, -9.8063307, -0.4521048, 0.4458299
5: -4.2563906, -3.6340449, -4.2563910, -3.6340427, -0.2980866, 0.2920210
6: -9.4212999, -8.5785799, -9.4213047, -8.5785809, -0.3743165, 0.3828144
7: -5.5672998, -4.7302275, -5.5673027, -4.7302265, -0.6084318, 0.6141286
8: -2.0258436, -1.2397361, -2.0258436, -1.2397342, -0.4582589, 0.4511800
9: -6.0475616, -5.4065599, -6.0475655, -5.4065599, -0.4313567, 0.4323800

Time for backsubstitution: 21.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1452

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344902
time: 3.29 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2359792
time: 7.25 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.7220135, -8.8412838, -9.7220192, -8.8412800, -0.5452893, 0.5496178
1: -9.3284960, -8.5760517, -9.3284969, -8.5760469, -0.4622166, 0.4642212
2: -0.3016465, 0.4025356, -0.3016474, 0.4025371, -0.4748542, 0.4780574
3: 4.1410470, 4.9639044, 4.1410475, 4.9639039, -0.6126685, 0.6136632
4: -10.6877213, -9.8063345, -10.6877155, -9.8063307, -0.4521074, 0.4458299
5: -4.2563910, -3.6340446, -4.2563910, -3.6340427, -0.2980864, 0.2920210
6: -9.4212990, -8.5785799, -9.4213047, -8.5785809, -0.3743165, 0.3828144
7: -5.5673003, -4.7302275, -5.5673027, -4.7302265, -0.6084323, 0.6141286
8: -2.0258465, -1.2397361, -2.0258436, -1.2397342, -0.4582589, 0.4511802
9: -6.0475616, -5.4065619, -6.0475655, -5.4065599, -0.4313564, 0.4323800

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 1452

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344902
time: 3.41 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2359799
time: 3.24 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.7220116, -8.8412819, -9.7220173, -8.8412838, -0.5452867, 0.5496182
1: -9.3284960, -8.5760489, -9.3284988, -8.5760517, -0.4622118, 0.4642262
2: -0.3016449, 0.4025385, -0.3016486, 0.4025355, -0.4748528, 0.4780605
3: 4.1410470, 4.9639034, 4.1410451, 4.9639053, -0.6126671, 0.6136637
4: -10.6877155, -9.8063335, -10.6877203, -9.8063316, -0.4521046, 0.4458351
5: -4.2563906, -3.6340449, -4.2563915, -3.6340427, -0.2980865, 0.2920210
6: -9.4212999, -8.5785799, -9.4213037, -8.5785809, -0.3743160, 0.3828149
7: -5.5672998, -4.7302275, -5.5673022, -4.7302275, -0.6084318, 0.6141286
8: -2.0258436, -1.2397361, -2.0258455, -1.2397337, -0.4582591, 0.4511821
9: -6.0475616, -5.4065599, -6.0475659, -5.4065628, -0.4313543, 0.4323800

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 1452

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344689
time: 3.43 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2349396
time: 3.50 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.7220135, -8.8412838, -9.7220173, -8.8412838, -0.5452900, 0.5496180
1: -9.3284960, -8.5760517, -9.3284988, -8.5760517, -0.4622138, 0.4642243
2: -0.3016465, 0.4025356, -0.3016486, 0.4025355, -0.4748540, 0.4780583
3: 4.1410470, 4.9639044, 4.1410451, 4.9639053, -0.6126671, 0.6136637
4: -10.6877213, -9.8063345, -10.6877203, -9.8063316, -0.4521060, 0.4458315
5: -4.2563910, -3.6340446, -4.2563915, -3.6340427, -0.2980864, 0.2920208
6: -9.4212990, -8.5785799, -9.4213037, -8.5785809, -0.3743162, 0.3828144
7: -5.5673003, -4.7302275, -5.5673022, -4.7302275, -0.6084323, 0.6141300
8: -2.0258465, -1.2397361, -2.0258455, -1.2397337, -0.4582593, 0.4511788
9: -6.0475616, -5.4065619, -6.0475659, -5.4065628, -0.4313564, 0.4323800

Time for backsubstitution: 21.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 1676
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 1452

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344689
time: 3.60 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2350621
time: 3.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.15 seconds
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344902
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2359792
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344902
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2359799
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344689
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2349396
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2346898, upper bound: 0.2344689
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.15
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2350621

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.7187529, -8.8580780, -9.7087450, -8.8865366, -0.4850264, 0.4937003
1: -9.3274450, -8.5826321, -9.3195095, -8.5913591, -0.4506783, 0.4446802
2: -0.2996963, 0.4007739, -0.2958541, 0.3959509, -0.4541001, 0.4641893
3: 4.1455383, 4.9639034, 4.1549673, 4.9639673, -0.5847836, 0.5876637
4: -10.6774836, -9.8068409, -10.6600266, -9.8247185, -0.4256794, 0.4146731
5: -4.2563782, -3.6352472, -4.2594647, -3.6387925, -0.2893503, 0.2844021
6: -9.4211445, -8.5812454, -9.4200859, -8.5864458, -0.3673790, 0.3717070
7: -5.5654459, -4.7385044, -5.5572348, -4.7521148, -0.5700748, 0.5736547
8: -2.0143085, -1.2403321, -1.9932208, -1.2592115, -0.4275510, 0.4201007
9: -6.0475593, -5.4223642, -6.0380783, -5.4520454, -0.3866010, 0.3901151

Time for backsubstitution: 20.97 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1452

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2354447, upper bound: 0.2354453
time: 3.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2354447, upper bound: 0.2354454
time: 3.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.7216225, -8.8472691, -9.7213240, -8.8526535, -0.4863231, 0.5468507
1: -9.3284683, -8.5782824, -9.3284445, -8.5804958, -0.4487298, 0.4613557
2: -0.2997338, 0.4025104, -0.2979283, 0.4024837, -0.4649212, 0.4623137
3: 4.1423573, 4.9639034, 4.1435432, 4.9639068, -0.5837626, 0.6165409
4: -10.6871243, -9.8063965, -10.6865273, -9.8064413, -0.4502904, 0.4175134
5: -4.2563877, -3.6348557, -4.2563844, -3.6356363, -0.2889781, 0.2851665
6: -9.4212971, -8.5794859, -9.4212971, -8.5803680, -0.3691866, 0.3784137
7: -5.5671396, -4.7347665, -5.5670156, -4.7389002, -0.5967717, 0.6123004
8: -2.0242348, -1.2398248, -2.0231762, -1.2398930, -0.4546556, 0.4104142
9: -6.0475626, -5.4102683, -6.0475645, -5.4139113, -0.3813968, 0.4243677

Time for backsubstitution: 21.00 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 2370
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 2880
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 421
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150
type: B, layer: 3, pos: 2874

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 668

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2348375, upper bound: 0.2312616
time: 3.26 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2348375, upper bound: 0.2348378
time: 3.28 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.7187529, -8.8580799, -9.7087450, -8.8865366, -0.4850268, 0.4936981
1: -9.3274450, -8.5826349, -9.3195095, -8.5913591, -0.4506795, 0.4446766
2: -0.2996972, 0.4007709, -0.2958541, 0.3959509, -0.4541006, 0.4641874
3: 4.1455364, 4.9639039, 4.1549673, 4.9639673, -0.5847840, 0.5876627
4: -10.6774883, -9.8068428, -10.6600266, -9.8247185, -0.4256845, 0.4146733
5: -4.2563782, -3.6352463, -4.2594647, -3.6387925, -0.2893504, 0.2844025
6: -9.4211445, -8.5812464, -9.4200859, -8.5864458, -0.3673785, 0.3717070
7: -5.5654469, -4.7385035, -5.5572348, -4.7521148, -0.5700748, 0.5736551
8: -2.0143094, -1.2403297, -1.9932208, -1.2592115, -0.4275534, 0.4201009
9: -6.0475597, -5.4223661, -6.0380783, -5.4520454, -0.3866007, 0.3901150

Time for backsubstitution: 21.02 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1452

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2344826, upper bound: 0.2344895
time: 5.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2344826, upper bound: 0.2344902
time: 3.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.7216244, -8.8472691, -9.7213240, -8.8526535, -0.4863234, 0.5468504
1: -9.3284693, -8.5782852, -9.3284445, -8.5804958, -0.4487331, 0.4613523
2: -0.2997339, 0.4025084, -0.2979283, 0.4024837, -0.4649227, 0.4623115
3: 4.1423550, 4.9639053, 4.1435432, 4.9639068, -0.5837655, 0.6165428
4: -10.6871290, -9.8063965, -10.6865273, -9.8064413, -0.4502912, 0.4175131
5: -4.2563891, -3.6348562, -4.2563844, -3.6356363, -0.2889781, 0.2851663
6: -9.4212990, -8.5794859, -9.4212971, -8.5803680, -0.3691864, 0.3784134
7: -5.5671411, -4.7347651, -5.5670156, -4.7389002, -0.5967722, 0.6123009
8: -2.0242376, -1.2398243, -2.0231762, -1.2398930, -0.4546559, 0.4104152
9: -6.0475616, -5.4102702, -6.0475645, -5.4139113, -0.3813989, 0.4243679

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 2370
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 2880
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 421
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150
type: B, layer: 3, pos: 2874

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 668

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2302432
time: 3.44 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2338680
time: 8.06 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -9.7187529, -8.8580780, -9.7087460, -8.8865376, -0.4850268, 0.4937015
1: -9.3274450, -8.5826321, -9.3195114, -8.5913601, -0.4506783, 0.4446833
2: -0.2996963, 0.4007739, -0.2958552, 0.3959480, -0.4540980, 0.4641898
3: 4.1455383, 4.9639034, 4.1549664, 4.9639668, -0.5847831, 0.5876637
4: -10.6774836, -9.8068409, -10.6600285, -9.8247194, -0.4256790, 0.4146779
5: -4.2563782, -3.6352472, -4.2594657, -3.6387939, -0.2893507, 0.2844030
6: -9.4211445, -8.5812454, -9.4200888, -8.5864468, -0.3673787, 0.3717074
7: -5.5654459, -4.7385044, -5.5572348, -4.7521157, -0.5700750, 0.5736547
8: -2.0143085, -1.2403321, -1.9932232, -1.2592106, -0.4275510, 0.4201009
9: -6.0475593, -5.4223642, -6.0380774, -5.4520464, -0.3866007, 0.3901148

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1452

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2344896, upper bound: 0.2344833
time: 3.25 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2344896, upper bound: 0.2344824
time: 3.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.7216225, -8.8472672, -9.7213240, -8.8526554, -0.4863210, 0.5468507
1: -9.3284683, -8.5782824, -9.3284473, -8.5804996, -0.4487298, 0.4613569
2: -0.2997338, 0.4025104, -0.2979291, 0.4024807, -0.4649208, 0.4623134
3: 4.1423573, 4.9639034, 4.1435413, 4.9639053, -0.5837636, 0.6165409
4: -10.6871243, -9.8063965, -10.6865311, -9.8064423, -0.4502902, 0.4175184
5: -4.2563877, -3.6348557, -4.2563848, -3.6356363, -0.2889785, 0.2851663
6: -9.4212971, -8.5794859, -9.4212952, -8.5803719, -0.3691864, 0.3784137
7: -5.5671396, -4.7347665, -5.5670161, -4.7389002, -0.5967722, 0.6122999
8: -2.0242352, -1.2398248, -2.0231786, -1.2398939, -0.4546566, 0.4104166
9: -6.0475626, -5.4102683, -6.0475655, -5.4139123, -0.3813970, 0.4243686

Time for backsubstitution: 20.88 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 2370
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 2880
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 421
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150
type: B, layer: 3, pos: 2874

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 668

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2338680, upper bound: 0.2293200
time: 3.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2338680, upper bound: 0.2327516
time: 3.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -9.7187529, -8.8580799, -9.7087460, -8.8865376, -0.4850273, 0.4937010
1: -9.3274450, -8.5826349, -9.3195114, -8.5913601, -0.4506798, 0.4446797
2: -0.2996972, 0.4007709, -0.2958552, 0.3959480, -0.4540987, 0.4641883
3: 4.1455364, 4.9639039, 4.1549664, 4.9639668, -0.5847826, 0.5876632
4: -10.6774883, -9.8068428, -10.6600285, -9.8247194, -0.4256809, 0.4146748
5: -4.2563782, -3.6352463, -4.2594657, -3.6387939, -0.2893510, 0.2844023
6: -9.4211445, -8.5812464, -9.4200888, -8.5864468, -0.3673785, 0.3717079
7: -5.5654469, -4.7385035, -5.5572348, -4.7521157, -0.5700760, 0.5736570
8: -2.0143094, -1.2403297, -1.9932232, -1.2592106, -0.4275503, 0.4201014
9: -6.0475597, -5.4223661, -6.0380774, -5.4520464, -0.3866007, 0.3901148

Time for backsubstitution: 21.06 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 1257
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 401
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: B, layer: 3, pos: 2495
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: A, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 2880
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2874
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1452

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2344682, upper bound: 0.2344689
time: 3.76 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2344682, upper bound: 0.2344689
time: 3.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.7216225, -8.8472700, -9.7213240, -8.8526554, -0.4863231, 0.5468514
1: -9.3284702, -8.5782843, -9.3284473, -8.5804996, -0.4487312, 0.4613557
2: -0.2997342, 0.4025090, -0.2979291, 0.4024807, -0.4649217, 0.4623125
3: 4.1423540, 4.9639049, 4.1435413, 4.9639053, -0.5837646, 0.6165414
4: -10.6871281, -9.8063955, -10.6865311, -9.8064423, -0.4502914, 0.4175148
5: -4.2563877, -3.6348565, -4.2563848, -3.6356363, -0.2889786, 0.2851667
6: -9.4212971, -8.5794849, -9.4212952, -8.5803719, -0.3691864, 0.3784142
7: -5.5671396, -4.7347651, -5.5670161, -4.7389002, -0.5967741, 0.6123009
8: -2.0242367, -1.2398243, -2.0231786, -1.2398939, -0.4546571, 0.4104142
9: -6.0475616, -5.4102702, -6.0475655, -5.4139123, -0.3813975, 0.4243689

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 668
type: B, layer: 3, pos: 668
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 1676
type: A, layer: 3, pos: 1676
type: A, layer: 3, pos: 2565
type: B, layer: 3, pos: 2565
type: B, layer: 3, pos: 2326
type: A, layer: 3, pos: 401
type: B, layer: 3, pos: 1257
type: B, layer: 3, pos: 401
type: A, layer: 3, pos: 1257
type: A, layer: 3, pos: 2326
type: B, layer: 3, pos: 2495
type: B, layer: 3, pos: 2606
type: A, layer: 3, pos: 2606
type: A, layer: 3, pos: 2370
type: A, layer: 3, pos: 2495
type: B, layer: 3, pos: 2370
type: B, layer: 3, pos: 1999
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1243
type: A, layer: 3, pos: 1243
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 1103
type: A, layer: 3, pos: 1999
type: A, layer: 3, pos: 1969
type: B, layer: 3, pos: 1969
type: B, layer: 3, pos: 1726
type: A, layer: 3, pos: 1726
type: A, layer: 3, pos: 2880
type: A, layer: 3, pos: 779
type: B, layer: 3, pos: 779
type: A, layer: 3, pos: 421
type: B, layer: 3, pos: 421
type: A, layer: 3, pos: 2874
type: B, layer: 3, pos: 2880
type: B, layer: 3, pos: 150
type: A, layer: 3, pos: 150
type: B, layer: 3, pos: 2874

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 668

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2294457
time: 3.48 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2329252
time: 3.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 28.17 seconds
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2354447, upper bound: 0.2354453
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2354447, upper bound: 0.2354454
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2348375, upper bound: 0.2312616
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2348375, upper bound: 0.2348378
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2344826, upper bound: 0.2344895
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2344826, upper bound: 0.2344902
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2302432
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2338680
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2344896, upper bound: 0.2344833
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2344896, upper bound: 0.2344824
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2338680, upper bound: 0.2293200
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2338680, upper bound: 0.2327516
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2344682, upper bound: 0.2344689
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2344682, upper bound: 0.2344689
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2294457
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 28.17
Output dim: 3, lower bound: -0.2327513, upper bound: 0.2329252

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.7087440, -8.8865366, -9.7087450, -8.8865366, -0.4540324, 0.4548361
1: -9.3195086, -8.5913591, -9.3195095, -8.5913591, -0.4390864, 0.4369092
2: -0.2958524, 0.3959499, -0.2958541, 0.3959509, -0.4452960, 0.4561324
3: 4.1549692, 4.9639659, 4.1549673, 4.9639673, -0.5692954, 0.5712786
4: -10.6600246, -9.8247185, -10.6600266, -9.8247185, -0.4061644, 0.3998933
5: -4.2594652, -3.6387944, -4.2594647, -3.6387925, -0.2849127, 0.2781718
6: -9.4200859, -8.5864468, -9.4200859, -8.5864458, -0.3612401, 0.3670485
7: -5.5572357, -4.7521157, -5.5572348, -4.7521148, -0.5433018, 0.5489054
8: -1.9932189, -1.2592115, -1.9932208, -1.2592115, -0.4082036, 0.4009209
9: -6.0380754, -5.4520459, -6.0380783, -5.4520454, -0.3624976, 0.3642831

Time for backsubstitution: 20.02 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.95 + 548.70 = 607.65 seconds
