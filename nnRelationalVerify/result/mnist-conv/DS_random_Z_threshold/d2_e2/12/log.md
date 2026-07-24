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
execution time: IAR + RelationalAnalysis = 25.02 + 34.17 = 59.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3290372, upper bound: 0.3290372

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6170
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 4582

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6170

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290367, upper bound: 0.3271944
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3271944, upper bound: 0.3290366
time: 3.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.94 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.94
Output dim: 6, lower bound: -0.3290367, upper bound: 0.3271944
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.94
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

Time for backsubstitution: 23.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 875

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4624

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290364, upper bound: 0.3268998
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3287421, upper bound: 0.3271942
time: 3.69 seconds

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

Time for backsubstitution: 23.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 844

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3246581, upper bound: 0.3257518
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3239095, upper bound: 0.3265004
time: 3.79 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.35
Output dim: 6, lower bound: -0.3290364, upper bound: 0.3268998
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.35
Output dim: 6, lower bound: -0.3287421, upper bound: 0.3271942
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.35
Output dim: 6, lower bound: -0.3246581, upper bound: 0.3257518
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.35
Output dim: 6, lower bound: -0.3239095, upper bound: 0.3265004

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6441734, 0.6359406
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5289979, 0.5462313
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4103765, 0.4152029
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4914074, 0.4964565
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7246089, 0.7287908
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5458949, 0.5496159
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4900806, 0.4889127
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7353806, 0.7498336
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5356627, 0.5342636
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6481457, 0.6338296

Time for backsubstitution: 23.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 5758

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 858

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261155, upper bound: 0.3236265
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3257631, upper bound: 0.3239790
time: 4.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6369636, 0.6431503
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5453506, 0.5298786
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4147177, 0.4108620
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4965644, 0.4912995
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7252793, 0.7281201
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5466442, 0.5488663
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4897120, 0.4892815
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7461686, 0.7390456
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5355282, 0.5343978
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6353700, 0.6466055

Time for backsubstitution: 23.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 844

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3262061, upper bound: 0.3239099
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3254575, upper bound: 0.3246578
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6646643, 0.6660383
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5954361, 0.5950904
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4282346, 0.4281995
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5118299, 0.5127891
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7266269, 0.7218735
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5461049, 0.5434573
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4915650, 0.4908147
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7823858, 0.7786551
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5348020, 0.5362833
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6847415, 0.6874399

Time for backsubstitution: 23.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4624

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4654

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3246546, upper bound: 0.3227704
time: 3.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3216767, upper bound: 0.3257490
time: 3.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6647401, 0.6656873
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5954361, 0.5945549
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4282351, 0.4277495
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5119891, 0.5119375
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7253852, 0.7221427
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5461049, 0.5431333
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4900155, 0.4911504
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7823200, 0.7786698
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5348020, 0.5360665
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6849928, 0.6862819

Time for backsubstitution: 23.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3232389, upper bound: 0.3248018
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3222265, upper bound: 0.3254507
time: 3.82 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3261155, upper bound: 0.3236265
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3257631, upper bound: 0.3239790
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3262061, upper bound: 0.3239099
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3254575, upper bound: 0.3246578
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3246546, upper bound: 0.3227704
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3216767, upper bound: 0.3257490
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3232389, upper bound: 0.3248018
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.45
Output dim: 6, lower bound: -0.3222265, upper bound: 0.3254507

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6441739, 0.6360598
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5289178, 0.5466239
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4101982, 0.4160770
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4912593, 0.4971820
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7253785, 0.7286341
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5459940, 0.5495954
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4907048, 0.4887850
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7354283, 0.7498336
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5358555, 0.5342240
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6480153, 0.6344669

Time for backsubstitution: 23.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 4654

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 486

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261137, upper bound: 0.3217614
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3242656, upper bound: 0.3236246
time: 3.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6441734, 0.6359410
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5289979, 0.5461512
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4103765, 0.4150240
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4914074, 0.4963086
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7244520, 0.7287908
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5458741, 0.5496159
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4899528, 0.4889127
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7353806, 0.7498336
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5356231, 0.5342636
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6481457, 0.6336992

Time for backsubstitution: 23.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5758

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3257549, upper bound: 0.3223536
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3241328, upper bound: 0.3239702
time: 3.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6368883, 0.6434259
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5453501, 0.5304141
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4147177, 0.4113119
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4964051, 0.4919918
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7262526, 0.7278516
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5466442, 0.5491900
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4909254, 0.4889457
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7462196, 0.7390313
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5355282, 0.5346148
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6351187, 0.6475122

Time for backsubstitution: 23.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 486

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3251563, upper bound: 0.3222261
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3245072, upper bound: 0.3232386
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6459608, 0.6528907
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5948319, 0.5946267
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4080594, 0.4130623
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4946628, 0.4999461
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7236857, 0.7179534
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5296440, 0.5311236
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4816539, 0.4833814
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7711315, 0.7636468
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5227973, 0.5272841
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6752288, 0.6747766

Time for backsubstitution: 23.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 4624
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3216671, upper bound: 0.3241163
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3200633, upper bound: 0.3257402
time: 3.45 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3261137, upper bound: 0.3217614
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3242656, upper bound: 0.3236246
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3257549, upper bound: 0.3223536
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3241328, upper bound: 0.3239702
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3251563, upper bound: 0.3222261
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3245072, upper bound: 0.3232386
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3216671, upper bound: 0.3241163
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.72
Output dim: 6, lower bound: -0.3200633, upper bound: 0.3257402

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6453018, 0.6369262
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5310268, 0.5493697
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4068389, 0.4135565
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4909863, 0.4968183
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7212226, 0.7266994
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5487721, 0.5543599
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4917676, 0.4901670
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7322960, 0.7480795
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5363212, 0.5342004
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6443686, 0.6291978

Time for backsubstitution: 23.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4654

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261101, upper bound: 0.3187758
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3231318, upper bound: 0.3217577
time: 3.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6401336, 0.6305015
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5288398, 0.5450408
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4086916, 0.4125628
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4942417, 0.5005295
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7229476, 0.7276623
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5451686, 0.5486760
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4932647, 0.4912244
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7353768, 0.7505224
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5359859, 0.5347831
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6470790, 0.6336105

Time for backsubstitution: 23.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 4654
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 5772

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3247090, upper bound: 0.3206802
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240579, upper bound: 0.3216887
time: 3.73 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.81
Output dim: 6, lower bound: -0.3261101, upper bound: 0.3187758
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.81
Output dim: 6, lower bound: -0.3231318, upper bound: 0.3217577
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.81
Output dim: 6, lower bound: -0.3247090, upper bound: 0.3206802
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.81
Output dim: 6, lower bound: -0.3240579, upper bound: 0.3216887

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6321542, 0.6182222
1: -7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5305636, 0.5487661
2: -7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.3917015, 0.3933806
3: -12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.4781435, 0.4796512
4: -0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7173009, 0.7237568
5: -7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5364382, 0.5378988
6: 0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4843345, 0.4802560
7: -4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7172856, 0.7368231
8: -0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5273218, 0.5221956
9: -5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6317055, 0.6196854

Time for backsubstitution: 23.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 5758
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 5772
type: DSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 81

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3250614, upper bound: 0.3187757
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3242339, upper bound: 0.3187756
time: 4.37 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 31.84 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 31.84
Output dim: 6, lower bound: -0.3250614, upper bound: 0.3187757
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 31.84
Output dim: 6, lower bound: -0.3242339, upper bound: 0.3187756

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 59.20 + 411.13 = 470.32 seconds
