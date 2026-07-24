## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.325746828


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6628447, 0.6628447)
1: (-7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5975900, 0.5975902)
2: (-7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4296961, 0.4296960)
3: (-12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5116768, 0.5116767)
4: (-0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7241158, 0.7241156)
5: (-7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5543370, 0.5543370)
6: (0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4935509, 0.4935508)
7: (-4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7933307, 0.7933311)
8: (-0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5398700, 0.5398700)
9: (-5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6888375, 0.6888375)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.86 + 34.23 = 57.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3290372, upper bound: 0.3290372

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 6170

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290367, upper bound: 0.3271944
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3271944, upper bound: 0.3290366
time: 3.82 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.53 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 6, lower bound: -0.3290367, upper bound: 0.3271944
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 6, lower bound: -0.3271944, upper bound: 0.3290366

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6657634, 0.6647403
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5945549, 0.5954361
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4277499, 0.4282351
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5120974, 0.5119894
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7221425, 0.7256544
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5431335, 0.5461051
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4911504, 0.4903513
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7786698, 0.7823348
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5360665, 0.5348020
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6865330, 0.6849928

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 5772

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290349, upper bound: 0.3251138
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269562, upper bound: 0.3271924
time: 4.05 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6647401, 0.6657634
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5954361, 0.5945549
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4282351, 0.4277499
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5119891, 0.5120972
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7256544, 0.7221427
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5461049, 0.5431335
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4903512, 0.4911504
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7823348, 0.7786698
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5348020, 0.5360665
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6849928, 0.6865330

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 5772

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3271926, upper bound: 0.3269567
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3251138, upper bound: 0.3290355
time: 3.74 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.05 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.05
Output dim: 6, lower bound: -0.3290349, upper bound: 0.3251138
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.05
Output dim: 6, lower bound: -0.3269562, upper bound: 0.3271924
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.05
Output dim: 6, lower bound: -0.3271926, upper bound: 0.3269567
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.05
Output dim: 6, lower bound: -0.3251138, upper bound: 0.3290355

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6664052, 0.6656041
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5828538, 0.5866611
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4166378, 0.4199853
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4977641, 0.5012423
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7138815, 0.7146370
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5417516, 0.5442629
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4842117, 0.4808747
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7786884, 0.7823484
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5333226, 0.5311444
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6755590, 0.6776845

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 4624

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290347, upper bound: 0.3248190
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3287404, upper bound: 0.3251136
time: 4.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6666274, 0.6653821
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5857801, 0.5837350
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4195002, 0.4171232
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5013502, 0.4976563
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7111254, 0.7173929
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5412915, 0.5447230
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4816740, 0.4834124
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7786837, 0.7823532
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5324090, 0.5320578
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6792247, 0.6740189

Time for backsubstitution: 23.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4624

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269560, upper bound: 0.3268979
time: 4.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3266614, upper bound: 0.3271922
time: 4.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6653824, 0.6666272
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5837350, 0.5857801
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4171232, 0.4195001
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4976563, 0.5013500
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7173929, 0.7111254
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5447230, 0.5412915
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4834125, 0.4816737
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7823534, 0.7786834
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5320580, 0.5324090
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6740189, 0.6792247

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4624

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3271924, upper bound: 0.3266613
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3268981, upper bound: 0.3269558
time: 3.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6656041, 0.6664052
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5866609, 0.5828540
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4199852, 0.4166380
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5012424, 0.4977641
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7146373, 0.7138815
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5442629, 0.5417516
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4808748, 0.4842114
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7823477, 0.7786882
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5311444, 0.5333223
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6776845, 0.6755590

Time for backsubstitution: 21.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4624

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3251135, upper bound: 0.3287404
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3248190, upper bound: 0.3290345
time: 3.98 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.80 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3290347, upper bound: 0.3248190
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3287404, upper bound: 0.3251136
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3269560, upper bound: 0.3268979
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3266614, upper bound: 0.3271922
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3271924, upper bound: 0.3266613
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3268981, upper bound: 0.3269558
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3251135, upper bound: 0.3287404
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.80
Output dim: 6, lower bound: -0.3248190, upper bound: 0.3290345

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6448152, 0.6368043
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5172951, 0.5374535
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.3992631, 0.4069499
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4770710, 0.4857042
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7163482, 0.7177742
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5445125, 0.5477734
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4831407, 0.4794350
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7353992, 0.7498474
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5329182, 0.5306056
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6371713, 0.6265209

Time for backsubstitution: 21.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290249, upper bound: 0.3231665
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3273964, upper bound: 0.3248110
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6376054, 0.6440139
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5336478, 0.5211021
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4036040, 0.4026104
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4822280, 0.4805491
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7170181, 0.7171037
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5452621, 0.5470238
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4827716, 0.4798037
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7461872, 0.7390594
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5327837, 0.5307400
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6243956, 0.6392968

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3287306, upper bound: 0.3234599
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3271035, upper bound: 0.3251058
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6450369, 0.6365824
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5202215, 0.5345287
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4021254, 0.4040892
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4806569, 0.4821202
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7135921, 0.7205298
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5440524, 0.5482335
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4806030, 0.4819727
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7353945, 0.7498522
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5320046, 0.5315192
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6408370, 0.6228554

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269480, upper bound: 0.3252632
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3252994, upper bound: 0.3268878
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6378272, 0.6437922
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5365727, 0.5181761
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4064647, 0.3997483
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4858119, 0.4769632
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7142630, 0.7198596
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5448020, 0.5474839
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4802339, 0.4823415
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7461824, 0.7390642
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5318704, 0.5316534
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6280613, 0.6356308

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3266533, upper bound: 0.3255562
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3250060, upper bound: 0.3271822
time: 3.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6437919, 0.6378274
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5181763, 0.5365727
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.3997483, 0.4064647
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4769633, 0.4858121
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7198596, 0.7142627
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5474842, 0.5448020
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4823415, 0.4802340
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7390642, 0.7461824
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5316536, 0.5318701
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6356311, 0.6280613

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3271822, upper bound: 0.3250059
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3255563, upper bound: 0.3266531
time: 8.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6365826, 0.6450372
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5345290, 0.5202211
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4040892, 0.4021252
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4821203, 0.4806570
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7205300, 0.7135923
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5482335, 0.5440524
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4819729, 0.4806029
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7498522, 0.7353945
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5315192, 0.5320046
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6228554, 0.6408370

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3268879, upper bound: 0.3252993
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3252633, upper bound: 0.3269478
time: 4.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6440141, 0.6376057
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5211022, 0.5336478
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4026104, 0.4036041
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4805491, 0.4822279
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7171040, 0.7170184
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5470240, 0.5452619
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4798038, 0.4827718
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7390594, 0.7461872
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5307400, 0.5327837
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6392965, 0.6243956

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3251057, upper bound: 0.3271040
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3234593, upper bound: 0.3287303
time: 4.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6368043, 0.6448152
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5374539, 0.5172951
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4069499, 0.3992631
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4857042, 0.4770709
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7177744, 0.7163479
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5477734, 0.5445125
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4794352, 0.4831406
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7498474, 0.7353992
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5306058, 0.5329180
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6265211, 0.6371713

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3248110, upper bound: 0.3273963
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3231659, upper bound: 0.3290247
time: 4.28 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3290249, upper bound: 0.3231665
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3273964, upper bound: 0.3248110
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3287306, upper bound: 0.3234599
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3271035, upper bound: 0.3251058
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3269480, upper bound: 0.3252632
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3252994, upper bound: 0.3268878
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3266533, upper bound: 0.3255562
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3250060, upper bound: 0.3271822
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3271822, upper bound: 0.3250059
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3255563, upper bound: 0.3266531
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3268879, upper bound: 0.3252993
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3252633, upper bound: 0.3269478
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3251057, upper bound: 0.3271040
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3234593, upper bound: 0.3287303
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3248110, upper bound: 0.3273963
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.21
Output dim: 6, lower bound: -0.3231659, upper bound: 0.3290247

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6407754, 0.6313651
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5171373, 0.5363433
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.3975780, 0.4044884
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4799056, 0.4899255
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7148438, 0.7166462
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5438070, 0.5468338
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4864528, 0.4817473
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7353954, 0.7505364
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5332813, 0.5311258
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6361046, 0.6264319

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 858

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261049, upper bound: 0.3218982
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3256599, upper bound: 0.3218990
time: 3.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6393759, 0.6327643
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5161848, 0.5372958
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.3968017, 0.4052647
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4812922, 0.4885387
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7152200, 0.7162700
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5435729, 0.5470679
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4854529, 0.4827473
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7360888, 0.7498431
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5334384, 0.5309689
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6370823, 0.6254544

Time for backsubstitution: 21.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 858

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3244859, upper bound: 0.3235192
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240367, upper bound: 0.3235198
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6335657, 0.6385746
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5334899, 0.5199919
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4019190, 0.4001490
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4850626, 0.4847704
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7155142, 0.7159758
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5445564, 0.5460842
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4860840, 0.4821161
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7461834, 0.7397490
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5331469, 0.5312603
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6233289, 0.6392078

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 858

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3258107, upper bound: 0.3221916
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3253656, upper bound: 0.3221925
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6321661, 0.6399739
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5325375, 0.5209442
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4011427, 0.4009253
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4864492, 0.4833837
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7158904, 0.7155995
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5443223, 0.5463183
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4850841, 0.4831160
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7468767, 0.7390552
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5333040, 0.5311031
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6243067, 0.6382301

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 858

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3241931, upper bound: 0.3238136
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3237437, upper bound: 0.3238145
time: 3.39 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3261049, upper bound: 0.3218982
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3256599, upper bound: 0.3218990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3244859, upper bound: 0.3235192
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3240367, upper bound: 0.3235198
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3258107, upper bound: 0.3221916
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3253656, upper bound: 0.3221925
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3241931, upper bound: 0.3238136
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 28.93
Output dim: 6, lower bound: -0.3237437, upper bound: 0.3238145
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3269480, upper bound: 0.3252632
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3252994, upper bound: 0.3268878
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3266533, upper bound: 0.3255562
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3250060, upper bound: 0.3271822
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3271822, upper bound: 0.3250059
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3255563, upper bound: 0.3266531
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3268879, upper bound: 0.3252993
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3252633, upper bound: 0.3269478
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3251057, upper bound: 0.3271040
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3234593, upper bound: 0.3287303
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3248110, upper bound: 0.3273963
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.93
Output dim: 6, lower bound: -0.3231659, upper bound: 0.3290247

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.09 + 546.33 = 603.42 seconds
