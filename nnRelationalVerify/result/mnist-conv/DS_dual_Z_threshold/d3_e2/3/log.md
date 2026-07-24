## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.36109017


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7289498, 0.7289498)
1: (-10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5894637, 0.5894637)
2: (-4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4779406, 0.4779406)
3: (-3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7333860, 0.7333860)
4: (-3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5551434, 0.5551434)
5: (-9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4376069, 0.4376069)
6: (-14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5916382, 0.5916383)
7: (3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7084303, 0.7084298)
8: (-6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5950758, 0.5950758)
9: (-1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4374404, 0.4374403)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.24 + 33.72 = 56.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4012108, upper bound: 0.4012113

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.49 seconds

### Candidate
type: DSZ, layer: 3, pos: 892

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3915799, upper bound: 0.3915807
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3915799, upper bound: 0.3915807
time: 4.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.66 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.66
Output dim: 7, lower bound: -0.3915799, upper bound: 0.3915807
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.66
Output dim: 7, lower bound: -0.3915799, upper bound: 0.3915807

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7346842, 0.7276735
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5887847, 0.5895684
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4819740, 0.4771290
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7322147, 0.7459633
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5547009, 0.5575404
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4375652, 0.4375619
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5896207, 0.6072099
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076807, 0.7056775
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5860381, 0.5938541
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4408572, 0.4373270

Time for backsubstitution: 8.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 2809

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3874433, upper bound: 0.3853221
time: 4.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3853218, upper bound: 0.3874433
time: 4.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7276735, 0.7289498
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5894637, 0.5887847
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4771290, 0.4779406
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7333860, 0.7322147
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5551434, 0.5547009
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4376069, 0.4375652
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5916382, 0.5896207
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7084303, 0.7076807
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5938540, 0.5950758
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4373271, 0.4374403

Time for backsubstitution: 8.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2809

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3874433, upper bound: 0.3853221
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3853218, upper bound: 0.3874433
time: 4.39 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 17.00 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 7, lower bound: -0.3874433, upper bound: 0.3853221
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 7, lower bound: -0.3853218, upper bound: 0.3874433
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 7, lower bound: -0.3874433, upper bound: 0.3853221
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 17.00
Output dim: 7, lower bound: -0.3853218, upper bound: 0.3874433

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7321751, 0.7254026
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5873301, 0.5867622
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4762113, 0.4674054
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7194805, 0.7295973
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5526390, 0.5564806
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4332306, 0.4341120
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5821700, 0.6008840
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7047977, 0.7029023
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5760584, 0.5840182
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4406058, 0.4371474

Time for backsubstitution: 8.51 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 1493

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858976, upper bound: 0.3821223
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3840127, upper bound: 0.3841017
time: 3.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7324133, 0.7276735
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5859785, 0.5895684
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4819740, 0.4713663
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7322147, 0.7332292
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5547009, 0.5554786
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4341153, 0.4375619
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5832946, 0.6072099
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7049060, 0.7056775
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5762024, 0.5938541
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4406775, 0.4373270

Time for backsubstitution: 8.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 1493

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3841013, upper bound: 0.3840130
time: 4.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3821218, upper bound: 0.3858984
time: 4.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7250047, 0.7266788
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5880089, 0.5859785
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4713663, 0.4682171
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7206807, 0.7158489
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5530820, 0.5537440
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4332724, 0.4341153
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5841875, 0.5832947
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7055469, 0.7049055
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5837135, 0.5852408
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4370757, 0.4372607

Time for backsubstitution: 9.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1493

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3858976, upper bound: 0.3821223
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3840127, upper bound: 0.3841017
time: 3.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7254026, 0.7289498
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5866575, 0.5887847
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4771290, 0.4721780
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7333860, 0.7194805
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5551434, 0.5526391
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4341570, 0.4375652
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5853124, 0.5896207
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7056546, 0.7076807
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5840182, 0.5950758
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4371475, 0.4374403

Time for backsubstitution: 8.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 1493

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3841013, upper bound: 0.3840130
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3821218, upper bound: 0.3858984
time: 4.01 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 17.55 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3858976, upper bound: 0.3821223
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3840127, upper bound: 0.3841017
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3841013, upper bound: 0.3840130
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3821218, upper bound: 0.3858984
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3858976, upper bound: 0.3821223
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3840127, upper bound: 0.3841017
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3841013, upper bound: 0.3840130
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.55
Output dim: 7, lower bound: -0.3821218, upper bound: 0.3858984

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7299364, 0.7229813
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5869603, 0.5860343
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4766375, 0.4673177
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7176301, 0.7269020
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5513406, 0.5520002
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4321775, 0.4329478
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5750724, 0.5949854
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7077713, 0.7056873
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5515504, 0.5581882
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4384614, 0.4344261

Time for backsubstitution: 9.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3829597, upper bound: 0.3793220
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3830354, upper bound: 0.3792359
time: 4.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7297540, 0.7231637
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5866022, 0.5863924
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4761237, 0.4681687
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7170866, 0.7277472
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5483623, 0.5551820
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4320666, 0.4330589
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5768343, 0.5937865
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076092, 0.7058761
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5550618, 0.5595102
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4378847, 0.4350029

Time for backsubstitution: 8.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3810863, upper bound: 0.3813763
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3811613, upper bound: 0.3812736
time: 4.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7301743, 0.7254601
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5856090, 0.5886903
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4826839, 0.4712785
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7308242, 0.7308352
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5534744, 0.5512019
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4330621, 0.4363052
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5761973, 0.6008779
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7078795, 0.7084534
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5516942, 0.5644227
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4385329, 0.4346005

Time for backsubstitution: 9.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3812732, upper bound: 0.3811622
time: 3.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3813759, upper bound: 0.3810871
time: 3.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7299919, 0.7256424
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5852509, 0.5890484
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4821701, 0.4717923
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7302806, 0.7313788
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5504961, 0.5541800
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4329511, 0.4364163
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5773960, 0.5996790
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076907, 0.7086422
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5503728, 0.5657448
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4379562, 0.4351773

Time for backsubstitution: 9.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3792354, upper bound: 0.3830354
time: 4.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3793215, upper bound: 0.3829606
time: 3.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7227659, 0.7242577
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5876393, 0.5852506
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4717923, 0.4681293
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7188301, 0.7131536
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5517828, 0.5492636
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4322194, 0.4329511
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5770897, 0.5773960
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7085204, 0.7076905
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5592053, 0.5594103
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4349314, 0.4345394

Time for backsubstitution: 9.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3829597, upper bound: 0.3793220
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3830354, upper bound: 0.3792359
time: 4.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7225835, 0.7244401
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5872812, 0.5856090
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4712785, 0.4689802
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7182865, 0.7139986
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5488045, 0.5524454
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4321084, 0.4330622
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5788516, 0.5761971
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7083583, 0.7078793
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5627167, 0.5607322
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4343547, 0.4351163

Time for backsubstitution: 9.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3810863, upper bound: 0.3813763
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3811613, upper bound: 0.3812736
time: 4.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7231636, 0.7267362
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5862875, 0.5879064
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4778390, 0.4720900
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7319956, 0.7170866
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5539169, 0.5483623
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4331040, 0.4363085
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5782143, 0.5832886
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7086282, 0.7104571
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5595102, 0.5656449
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4350029, 0.4347136

Time for backsubstitution: 9.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3812732, upper bound: 0.3811622
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3813759, upper bound: 0.3810871
time: 3.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7229815, 0.7269186
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5859294, 0.5882647
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4773252, 0.4726038
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7314517, 0.7176301
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5509386, 0.5513406
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4329929, 0.4364195
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5794133, 0.5820897
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7084394, 0.7106454
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5581882, 0.5669668
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4344262, 0.4352905

Time for backsubstitution: 9.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3792354, upper bound: 0.3830355
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3793215, upper bound: 0.3829606
time: 3.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 16.79 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3829597, upper bound: 0.3793220
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3830354, upper bound: 0.3792359
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3810863, upper bound: 0.3813763
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3811613, upper bound: 0.3812736
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3812732, upper bound: 0.3811622
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3813759, upper bound: 0.3810871
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3792354, upper bound: 0.3830354
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3793215, upper bound: 0.3829606
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3829597, upper bound: 0.3793220
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3830354, upper bound: 0.3792359
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3810863, upper bound: 0.3813763
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3811613, upper bound: 0.3812736
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3812732, upper bound: 0.3811622
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3813759, upper bound: 0.3810871
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3792354, upper bound: 0.3830355
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.79
Output dim: 7, lower bound: -0.3793215, upper bound: 0.3829606

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7249026, 0.7183241
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5695505, 0.5700696
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4730791, 0.4628649
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7110572, 0.7189622
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5511577, 0.5518765
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4313040, 0.4323591
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5612421, 0.5808674
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7077703, 0.7056861
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5318215, 0.5389318
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4335703, 0.4303652

Time for backsubstitution: 9.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3821115, upper bound: 0.3751354
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3788778, upper bound: 0.3783655
time: 4.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7252793, 0.7179474
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5709953, 0.5686243
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4721845, 0.4637594
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7096906, 0.7203290
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5512168, 0.5518174
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4315889, 0.4320742
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5609543, 0.5811553
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7077703, 0.7056861
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5322940, 0.5384595
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4344006, 0.4295357

Time for backsubstitution: 9.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3821923, upper bound: 0.3750457
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3789589, upper bound: 0.3782771
time: 4.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7247200, 0.7185065
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5691924, 0.5704277
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4725653, 0.4637159
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7105134, 0.7198074
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5481794, 0.5550585
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4311930, 0.4324701
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5630043, 0.5796685
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076082, 0.7058749
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5353329, 0.5402539
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4329942, 0.4309421

Time for backsubstitution: 9.15 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3802399, upper bound: 0.3771698
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770099, upper bound: 0.3803818
time: 3.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7250967, 0.7181296
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5706372, 0.5689828
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4716707, 0.4646103
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7091467, 0.7211740
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5482385, 0.5549994
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4314779, 0.4321852
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5627162, 0.5799564
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076082, 0.7058749
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5358055, 0.5397816
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4338237, 0.4301118

Time for backsubstitution: 9.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3803189, upper bound: 0.3770620
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770900, upper bound: 0.3802746
time: 3.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7251406, 0.7208029
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5681992, 0.5727251
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4791256, 0.4668257
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7242513, 0.7228956
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5532918, 0.5510783
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4321885, 0.4357163
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5623672, 0.5867599
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7078781, 0.7084527
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5319655, 0.5451664
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4336418, 0.4305395

Time for backsubstitution: 9.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3802741, upper bound: 0.3770899
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770621, upper bound: 0.3803188
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7255173, 0.7204261
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5696440, 0.5712798
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4782311, 0.4677203
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7228847, 0.7242622
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5533509, 0.5510192
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4324735, 0.4354314
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5620792, 0.5870479
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7078781, 0.7084527
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5324380, 0.5446941
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4344721, 0.4297100

Time for backsubstitution: 9.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3803814, upper bound: 0.3770098
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3771694, upper bound: 0.3802399
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7249579, 0.7209852
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5678406, 0.5730832
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4786118, 0.4673395
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7237077, 0.7234392
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5503135, 0.5540564
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4320775, 0.4358273
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5635660, 0.5855612
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076893, 0.7086420
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5306442, 0.5464884
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4330658, 0.4311163

Time for backsubstitution: 9.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3782766, upper bound: 0.3789588
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3750453, upper bound: 0.3821923
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7253346, 0.7206084
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5692859, 0.5716383
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4777173, 0.4682341
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7223411, 0.7248058
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5503726, 0.5539973
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4323624, 0.4355425
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5632780, 0.5858490
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076893, 0.7086415
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5311167, 0.5460161
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4338952, 0.4302860

Time for backsubstitution: 9.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3783650, upper bound: 0.3788786
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3751349, upper bound: 0.3821115
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7177320, 0.7196001
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5702295, 0.5692856
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4682341, 0.4636763
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7122571, 0.7052140
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5516007, 0.5491400
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4313456, 0.4323624
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5632594, 0.5632781
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7085195, 0.7076898
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5394766, 0.5401540
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4300402, 0.4304785

Time for backsubstitution: 9.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3821115, upper bound: 0.3751354
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3788778, upper bound: 0.3783655
time: 3.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7181087, 0.7192234
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5716743, 0.5678408
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4673395, 0.4645709
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7108905, 0.7065806
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5516598, 0.5490808
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4316305, 0.4320775
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5629716, 0.5635661
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7085195, 0.7076893
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5399487, 0.5396817
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4308705, 0.4296490

Time for backsubstitution: 9.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3821923, upper bound: 0.3750458
time: 4.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3789589, upper bound: 0.3782771
time: 3.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7175493, 0.7197825
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5698709, 0.5696437
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4677203, 0.4645274
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7117136, 0.7060590
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5486224, 0.5523218
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4312346, 0.4324734
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5650213, 0.5620792
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7083573, 0.7078781
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5429881, 0.5414760
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4294642, 0.4310552

Time for backsubstitution: 9.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3802399, upper bound: 0.3771698
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770099, upper bound: 0.3803818
time: 3.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7179265, 0.7194058
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5713162, 0.5681989
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4668257, 0.4654218
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7103469, 0.7074256
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5486815, 0.5522627
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4315195, 0.4321885
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5647335, 0.5623672
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7083578, 0.7078781
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5434606, 0.5410037
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4302937, 0.4302250

Time for backsubstitution: 9.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3803189, upper bound: 0.3770620
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770900, upper bound: 0.3802746
time: 3.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7181296, 0.7220787
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5688777, 0.5719411
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4742806, 0.4676372
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7254226, 0.7091470
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5537341, 0.5482385
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4322302, 0.4357196
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5643842, 0.5691707
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7086277, 0.7104564
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5397813, 0.5463884
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4301118, 0.4306527

Time for backsubstitution: 9.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3802741, upper bound: 0.3770899
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770621, upper bound: 0.3803188
time: 3.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7185063, 0.7217020
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5703225, 0.5704963
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4733860, 0.4685316
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7240560, 0.7105136
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5537932, 0.5481794
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4325151, 0.4354347
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5640962, 0.5694587
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7086277, 0.7104564
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5402539, 0.5459161
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4309421, 0.4298233

Time for backsubstitution: 9.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3803814, upper bound: 0.3770098
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3771694, upper bound: 0.3802399
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7179475, 0.7222611
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5685191, 0.5722997
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4737668, 0.4681510
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7248788, 0.7096906
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5507557, 0.5512170
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4321191, 0.4358307
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5655832, 0.5679718
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7084389, 0.7106447
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5384595, 0.5477104
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4295357, 0.4312295

Time for backsubstitution: 9.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3782766, upper bound: 0.3789588
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3750453, upper bound: 0.3821923
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7183242, 0.7218843
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5699644, 0.5708549
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4728723, 0.4690454
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7235122, 0.7110572
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5508149, 0.5511578
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4324040, 0.4355457
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5652952, 0.5682598
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7084389, 0.7106447
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5389321, 0.5472381
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4303652, 0.4303993

Time for backsubstitution: 9.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3783650, upper bound: 0.3788786
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3751349, upper bound: 0.3821115
time: 3.47 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 16.79 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3821115, upper bound: 0.3751354
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3788778, upper bound: 0.3783655
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3821923, upper bound: 0.3750457
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3789589, upper bound: 0.3782771
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3802399, upper bound: 0.3771698
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3770099, upper bound: 0.3803818
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3803189, upper bound: 0.3770620
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3770900, upper bound: 0.3802746
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3802741, upper bound: 0.3770899
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3770621, upper bound: 0.3803188
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3803814, upper bound: 0.3770098
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3771694, upper bound: 0.3802399
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3782766, upper bound: 0.3789588
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3750453, upper bound: 0.3821923
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3783650, upper bound: 0.3788786
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3751349, upper bound: 0.3821115
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3821115, upper bound: 0.3751354
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3788778, upper bound: 0.3783655
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3821923, upper bound: 0.3750458
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3789589, upper bound: 0.3782771
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3802399, upper bound: 0.3771698
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3770099, upper bound: 0.3803818
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3803189, upper bound: 0.3770620
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3770900, upper bound: 0.3802746
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3802741, upper bound: 0.3770899
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3770621, upper bound: 0.3803188
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3803814, upper bound: 0.3770098
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3771694, upper bound: 0.3802399
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3782766, upper bound: 0.3789588
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3750453, upper bound: 0.3821923
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3783650, upper bound: 0.3788786
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.79
Output dim: 7, lower bound: -0.3751349, upper bound: 0.3821115

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7255683, 0.7178407
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5585372, 0.5673927
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4747562, 0.4627491
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7062809, 0.7168229
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5276275, 0.5127314
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4285618, 0.4315250
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5373731, 0.5504556
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7076516, 0.7054403
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5278280, 0.5374167
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4338392, 0.4303504

Time for backsubstitution: 9.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 2803

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.96 + 547.29 = 604.25 seconds
