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
execution time: IAR + RelationalAnalysis = 22.35 + 35.39 = 57.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2393172, upper bound: 0.2393173

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387022
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387020, upper bound: 0.2387021
time: 7.90 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.21 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.21
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387022
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.21
Output dim: 3, lower bound: -0.2387020, upper bound: 0.2387021

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5519714, 0.5510931
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4619343, 0.4622202
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4733138, 0.4748595
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6136699, 0.6126781
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4458368, 0.4449413
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2920229, 0.2907050
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3731089, 0.3743186
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6141367, 0.6133881
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4511859, 0.4498456
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4308026, 0.4313648

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2373800
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2373800
time: 2.87 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5510931, 0.5519712
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4622202, 0.4619343
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4748592, 0.4733136
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6126781, 0.6136703
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4449413, 0.4458368
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2907050, 0.2920227
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3743186, 0.3731089
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6133881, 0.6141367
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4498456, 0.4511862
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4313648, 0.4308026

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373794, upper bound: 0.2373968
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373794, upper bound: 0.2375029
time: 3.07 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.85 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.85
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2373800
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.85
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2373800
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.85
Output dim: 3, lower bound: -0.2373794, upper bound: 0.2373968
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.85
Output dim: 3, lower bound: -0.2373794, upper bound: 0.2375029

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5519714, 0.5510931
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4619346, 0.4622204
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4733138, 0.4748595
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6136708, 0.6126785
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4458365, 0.4449406
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2920227, 0.2907047
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3731089, 0.3743186
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6141367, 0.6133876
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4511857, 0.4498451
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4308028, 0.4313645

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 2495

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2368890, upper bound: 0.2369350
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2370662, upper bound: 0.2367603
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5519710, 0.5510931
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4619343, 0.4622202
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4733138, 0.4748595
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6136703, 0.6126781
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4458361, 0.4449413
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2920225, 0.2907050
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3731089, 0.3743186
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6141367, 0.6133881
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4511855, 0.4498456
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4308023, 0.4313648

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 2565

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 401

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2363332, upper bound: 0.2348520
time: 3.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2348682, upper bound: 0.2363170
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5510931, 0.5519712
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4622202, 0.4619343
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4748592, 0.4733136
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6126781, 0.6136703
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4449410, 0.4458361
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2907045, 0.2920227
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3743186, 0.3731089
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6133876, 0.6141367
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4498453, 0.4511855
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4313645, 0.4308023

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 1676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3104

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2366882, upper bound: 0.2326367
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2326185, upper bound: 0.2367057
time: 3.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5510931, 0.5519712
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4622204, 0.4619343
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4748592, 0.4733136
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6126781, 0.6136703
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4449406, 0.4458368
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2907047, 0.2920227
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3743186, 0.3731089
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6133876, 0.6141367
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4498448, 0.4511862
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4313645, 0.4308026

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 401

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2460

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2347182, upper bound: 0.2362677
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2361619, upper bound: 0.2348251
time: 3.13 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.47 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2368890, upper bound: 0.2369350
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2370662, upper bound: 0.2367603
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2363332, upper bound: 0.2348520
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2348682, upper bound: 0.2363170
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2366882, upper bound: 0.2326367
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2326185, upper bound: 0.2367057
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2347182, upper bound: 0.2362677
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.47
Output dim: 3, lower bound: -0.2361619, upper bound: 0.2348251

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5507662, 0.5494344
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4613099, 0.4603918
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4722803, 0.4735112
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6135483, 0.6127121
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4513845, 0.4474435
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2900124, 0.2892861
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3698487, 0.3706429
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6136918, 0.6131020
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4487755, 0.4475746
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4212580, 0.4254613

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 401

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2326

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2320278, upper bound: 0.2361173
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2360706, upper bound: 0.2320773
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5503128, 0.5498879
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4601059, 0.4615958
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4719656, 0.4738259
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6137042, 0.6125557
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4483395, 0.4504886
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2906041, 0.2886944
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3694329, 0.3710587
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6138506, 0.6129427
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4489155, 0.4474347
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4248991, 0.4218199

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2606

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2366683, upper bound: 0.2308048
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2311069, upper bound: 0.2363624
time: 3.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5505438, 0.5498726
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4626443, 0.4630439
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4722786, 0.4743669
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6092534, 0.6081166
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4417613, 0.4410470
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2741147, 0.2732487
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3648887, 0.3667984
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6124220, 0.6104641
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4445086, 0.4430158
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4077368, 0.4098442

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1515

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2460

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2332005, upper bound: 0.2333585
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346979, upper bound: 0.2331620
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5507507, 0.5496659
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4627578, 0.4629307
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4728217, 0.4738238
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6091089, 0.6082611
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4419420, 0.4408662
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2745665, 0.2727971
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3655887, 0.3660989
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6112127, 0.6116729
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4443562, 0.4431679
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4092820, 0.4082990

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 2326

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 421

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2348648, upper bound: 0.2363147
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2348663, upper bound: 0.2363135
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5422342, 0.5446510
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4492338, 0.4497795
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4737027, 0.4721100
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6047459, 0.6031568
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4094467, 0.4161661
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2811430, 0.2831013
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3676822, 0.3670931
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6065283, 0.6044359
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4331017, 0.4359086
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4125006, 0.4129629

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1103

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1969

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2365416, upper bound: 0.2298077
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2338482, upper bound: 0.2324903
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5437729, 0.5431120
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4500656, 0.4489479
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4736557, 0.4721570
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6021647, 0.6057384
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4152708, 0.4103420
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2817836, 0.2824609
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3683033, 0.3664720
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6036868, 0.6072774
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4345684, 0.4344416
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4135249, 0.4119387

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 150

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2326185, upper bound: 0.2362268
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2321442, upper bound: 0.2367056
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5508780, 0.5515828
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4529114, 0.4475741
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4651957, 0.4642067
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6119857, 0.6132331
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4388306, 0.4390414
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2849033, 0.2865603
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3728280, 0.3712955
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6130533, 0.6138573
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4313474, 0.4318423
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4153755, 0.4176886

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1969

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2342910, upper bound: 0.2357935
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2342885, upper bound: 0.2356963
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5507050, 0.5517561
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4478600, 0.4526255
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4657526, 0.4636500
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6122413, 0.6129775
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4381454, 0.4397266
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2852423, 0.2862217
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3725057, 0.3716178
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6131091, 0.6138020
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4305010, 0.4326887
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4182508, 0.4148130

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2495

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349456, upper bound: 0.2336750
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2350119, upper bound: 0.2336089
time: 3.25 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2320278, upper bound: 0.2361173
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2360706, upper bound: 0.2320773
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2366683, upper bound: 0.2308048
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2311069, upper bound: 0.2363624
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2332005, upper bound: 0.2333585
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2346979, upper bound: 0.2331620
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2348648, upper bound: 0.2363147
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2348663, upper bound: 0.2363135
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2365416, upper bound: 0.2298077
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2338482, upper bound: 0.2324903
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2326185, upper bound: 0.2362268
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2321442, upper bound: 0.2367056
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2342910, upper bound: 0.2357935
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2342885, upper bound: 0.2356963
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2349456, upper bound: 0.2336750
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.06
Output dim: 3, lower bound: -0.2350119, upper bound: 0.2336089

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5519092, 0.5510232
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4625781, 0.4625816
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4711144, 0.4730093
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6045752, 0.6054516
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4356759, 0.4338026
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2892642, 0.2882757
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3635228, 0.3635049
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6054654, 0.6043286
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4238470, 0.4192367
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4197576, 0.4182055

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 2565

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2290711, upper bound: 0.2336202
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2295056, upper bound: 0.2334050
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5519016, 0.5510309
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4622960, 0.4628639
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4714639, 0.4726601
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6064439, 0.6035829
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4346986, 0.4347799
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2895937, 0.2879462
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3622949, 0.3647327
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6050773, 0.6047163
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4205775, 0.4225059
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4176433, 0.4203196

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 1999

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1726

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2358344, upper bound: 0.2311621
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2355264, upper bound: 0.2317903
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5521994, 0.5513663
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4605935, 0.4611819
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4702780, 0.4714966
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6202354, 0.6178827
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4422045, 0.4418447
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2902482, 0.2893221
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3724082, 0.3741462
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6248183, 0.6234651
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4517527, 0.4505925
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4280374, 0.4290233

Time for backsubstitution: 22.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 1257

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2365666, upper bound: 0.2291101
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2350856, upper bound: 0.2306367
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5522447, 0.5513210
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4608960, 0.4608793
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4699512, 0.4718237
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6188750, 0.6192431
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4427407, 0.4413085
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2906401, 0.2889302
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3729365, 0.3736179
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6242137, 0.6240697
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4519334, 0.4504118
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4284613, 0.4285994

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 150
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 1676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2874

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 3104

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2303932, upper bound: 0.2319512
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2258432, upper bound: 0.2354674
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.5515828, 0.5508780
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4475741, 0.4529116
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4642067, 0.4651957
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6132336, 0.6119857
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4390409, 0.4388311
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2865603, 0.2849038
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3712955, 0.3728280
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6138577, 0.6130533
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4318414, 0.4313481
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.4176886, 0.4153755

Time for backsubstitution: 21.62 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.74 + 553.53 = 611.27 seconds
