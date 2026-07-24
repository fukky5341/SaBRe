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
execution time: IAR + RelationalAnalysis = 22.53 + 36.31 = 58.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2393172, upper bound: 0.2393173

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387022
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2387020, upper bound: 0.2387021
time: 8.22 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.89 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.89
Output dim: 3, lower bound: -0.2393160, upper bound: 0.2387022
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.89
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

Time for backsubstitution: 20.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2373800
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2373800
time: 3.11 seconds

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

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 106

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 106

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373794, upper bound: 0.2373968
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2373794, upper bound: 0.2375029
time: 3.20 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.12
Output dim: 3, lower bound: -0.2375023, upper bound: 0.2373800
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.12
Output dim: 3, lower bound: -0.2373962, upper bound: 0.2373800
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.12
Output dim: 3, lower bound: -0.2373794, upper bound: 0.2373968
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.12
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

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346857, upper bound: 0.2349201
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2350618, upper bound: 0.2346864
time: 3.44 seconds

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

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346857, upper bound: 0.2349201
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2346864
time: 3.21 seconds

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

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346858, upper bound: 0.2349403
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349194, upper bound: 0.2346864
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

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2346858, upper bound: 0.2350618
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2349194, upper bound: 0.2346864
time: 3.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.66 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2346857, upper bound: 0.2349201
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2350618, upper bound: 0.2346864
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2346857, upper bound: 0.2349201
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2349402, upper bound: 0.2346864
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2346858, upper bound: 0.2349403
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2349194, upper bound: 0.2346864
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2346858, upper bound: 0.2350618
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.66
Output dim: 3, lower bound: -0.2349194, upper bound: 0.2346864

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4868245, 0.4852371
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4468455, 0.4453700
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4611790, 0.4626615
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6215315, 0.6204071
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4162610, 0.4164233
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2863563, 0.2867630
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3679171, 0.3686831
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6019549, 0.6027241
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4064236, 0.4088833
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3776329, 0.3747888

Time for backsubstitution: 20.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2321110
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2298965
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4861155, 0.4885957
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4450841, 0.4488282
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4624617, 0.4627247
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6240029, 0.6205392
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4185495, 0.4153650
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2880810, 0.2856998
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3674729, 0.3699572
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6034727, 0.6016722
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4122066, 0.4050827
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3742268, 0.3816862

Time for backsubstitution: 21.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2300427, upper bound: 0.2318797
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2322502, upper bound: 0.2296666
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4868243, 0.4852374
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4468455, 0.4453700
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4611788, 0.4626615
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6215315, 0.6204076
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4162605, 0.4164240
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2863561, 0.2867631
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3679171, 0.3686831
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6019549, 0.6027246
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4064231, 0.4088833
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3776326, 0.3747890

Time for backsubstitution: 21.58 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2321110
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2298965
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4861152, 0.4885960
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4450839, 0.4488282
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4624617, 0.4627249
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6240029, 0.6205397
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4185491, 0.4153655
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2880808, 0.2856998
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3674729, 0.3699572
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6034727, 0.6016726
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4122062, 0.4050829
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3742266, 0.3816862

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2299177, upper bound: 0.2318797
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2321305, upper bound: 0.2296665
time: 3.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4885955, 0.4861152
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4488280, 0.4450839
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4627244, 0.4624617
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6205392, 0.6240029
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4153652, 0.4185491
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2856997, 0.2880808
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3699572, 0.3674729
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6016722, 0.6034727
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4050832, 0.4122062
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3816860, 0.3742266

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2321311
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2299182
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4852374, 0.4868243
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4453697, 0.4468455
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4626613, 0.4611790
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6204071, 0.6215310
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4164236, 0.4162605
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2867628, 0.2863561
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3686831, 0.3679171
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6027241, 0.6019549
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4088836, 0.4064231
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3747888, 0.3776326

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2298959, upper bound: 0.2318797
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2321104, upper bound: 0.2296665
time: 3.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4885957, 0.4861155
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4488282, 0.4450839
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4627247, 0.4624619
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6205392, 0.6240029
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4153650, 0.4185495
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2857000, 0.2880809
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3699572, 0.3674729
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6016722, 0.6034732
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4050827, 0.4122062
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3816862, 0.3742268

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2322509
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2300434
time: 3.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -9.7220268, -8.8412800, -9.7220268, -8.8412800, -0.4852371, 0.4868245
1: -9.3285027, -8.5760479, -9.3285027, -8.5760479, -0.4453700, 0.4468455
2: -0.3016519, 0.4025378, -0.3016519, 0.4025378, -0.4626613, 0.4611790
3: 4.1410451, 4.9639072, 4.1410451, 4.9639072, -0.6204076, 0.6215320
4: -10.6877155, -9.8063240, -10.6877155, -9.8063240, -0.4164233, 0.4162610
5: -4.2563906, -3.6340394, -4.2563906, -3.6340394, -0.2867631, 0.2863561
6: -9.4213066, -8.5785809, -9.4213066, -8.5785809, -0.3686831, 0.3679171
7: -5.5673070, -4.7302270, -5.5673070, -4.7302270, -0.6027241, 0.6019554
8: -2.0258446, -1.2397323, -2.0258446, -1.2397323, -0.4088833, 0.4064233
9: -6.0475712, -5.4065585, -6.0475712, -5.4065585, -0.3747888, 0.3776326

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1676
type: DSZ, layer: 3, pos: 2565
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2495
type: DSZ, layer: 3, pos: 1257
type: DSZ, layer: 3, pos: 401
type: DSZ, layer: 3, pos: 668
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2880
type: DSZ, layer: 3, pos: 1999
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1969
type: DSZ, layer: 3, pos: 2874
type: DSZ, layer: 3, pos: 2606
type: DSZ, layer: 3, pos: 779
type: DSZ, layer: 3, pos: 1726
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 150

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1676

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2298959, upper bound: 0.2318797
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2321104, upper bound: 0.2296665
time: 3.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.35 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2321110
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2298965
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2300427, upper bound: 0.2318797
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2322502, upper bound: 0.2296666
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2321110
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2298965
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2299177, upper bound: 0.2318797
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2321305, upper bound: 0.2296665
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2321311
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2299182
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2298959, upper bound: 0.2318797
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2321104, upper bound: 0.2296665
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2296665, upper bound: 0.2322509
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2318791, upper bound: 0.2300434
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2298959, upper bound: 0.2318797
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.35
Output dim: 3, lower bound: -0.2321104, upper bound: 0.2296665

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 58.84 + 413.75 = 472.59 seconds
