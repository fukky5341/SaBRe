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
execution time: IAR + RelationalAnalysis = 22.48 + 35.63 = 58.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2393172, upper bound: 0.2393173

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 106

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 511

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387026
time: 4.04 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2393161
time: 5.41 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.67
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387026
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.67
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

Time for backsubstitution: 20.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 511

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387022
time: 5.88 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387023
time: 3.61 seconds

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

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 511

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2393161
time: 4.67 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387020, upper bound: 0.2393161
time: 3.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.03 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.03
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387022
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.03
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2387023
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.03
Output dim: 3, lower bound: -0.2387019, upper bound: 0.2393161
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.03
Output dim: 3, lower bound: -0.2387020, upper bound: 0.2393161

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

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

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

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -9.7219696, -8.8417940, -9.7005424, -8.8567333, -0.5258753, 0.5232213
1: -9.3284864, -8.5762644, -9.3144932, -8.5820627, -0.4574988, 0.4493260
2: -0.3015026, 0.4025243, -0.2892741, 0.3964887, -0.4701345, 0.4620645
3: 4.1411657, 4.9639053, 4.1511898, 4.9600511, -0.5970860, 0.5998037
4: -10.6876116, -9.8063412, -10.6752739, -9.8232450, -0.4346561, 0.4362993
5: -4.2563901, -3.6341074, -4.2518725, -3.6431301, -0.2866756, 0.2905817
6: -9.4212999, -8.5786591, -9.4135218, -8.5854979, -0.3749638, 0.3729916
7: -5.5672779, -4.7306304, -5.5471950, -4.7415795, -0.5973821, 0.5868492
8: -2.0255909, -1.2397461, -2.0102525, -1.2585731, -0.4391894, 0.4382081
9: -6.0475607, -5.4069166, -6.0299177, -5.4221191, -0.4107640, 0.4127088

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.7220135, -8.8412838, -9.7220135, -8.8412838, -0.5519605, 0.5496171
1: -9.3284950, -8.5760498, -9.3284950, -8.5760498, -0.4622149, 0.4622149
2: -0.3016455, 0.4025380, -0.3016455, 0.4025380, -0.4748521, 0.4748518
3: 4.1410475, 4.9639044, 4.1410475, 4.9639044, -0.6136613, 0.6136611
4: -10.6877155, -9.8063335, -10.6877155, -9.8063335, -0.4458303, 0.4458303
5: -4.2563906, -3.6340444, -4.2563906, -3.6340444, -0.2920201, 0.2920203
6: -9.4213018, -8.5785809, -9.4213018, -8.5785809, -0.3743162, 0.3743160
7: -5.5672989, -4.7302260, -5.5672989, -4.7302260, -0.6141253, 0.6141253
8: -2.0258436, -1.2397356, -2.0258436, -1.2397356, -0.4511800, 0.4511800
9: -6.0475612, -5.4065604, -6.0475612, -5.4065604, -0.4302807, 0.4313560

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 106

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.10 + 181.21 = 239.31 seconds
