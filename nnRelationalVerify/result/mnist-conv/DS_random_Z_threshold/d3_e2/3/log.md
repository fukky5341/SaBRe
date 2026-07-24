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
execution time: IAR + RelationalAnalysis = 23.98 + 33.59 = 57.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4012108, upper bound: 0.4012113

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1493
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1493

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4002947, upper bound: 0.3982608
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3982611, upper bound: 0.4002953
time: 3.40 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.67 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.67
Output dim: 7, lower bound: -0.4002947, upper bound: 0.3982608
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.67
Output dim: 7, lower bound: -0.3982611, upper bound: 0.4002953

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7269185, 0.7267362
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5889432, 0.5885851
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4786503, 0.4781365
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7319956, 0.7314517
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5539169, 0.5509386
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4364613, 0.4363503
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5841069, 0.5853057
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7113943, 0.7112057
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5669668, 0.5656449
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4352905, 0.4347136

Time for backsubstitution: 8.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2867

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3941637, upper bound: 0.3920048
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3941637, upper bound: 0.3920048
time: 3.54 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7267363, 0.7269186
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5885851, 0.5889432
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4781365, 0.4786503
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7314517, 0.7319956
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5509386, 0.5539169
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4363503, 0.4364614
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5853057, 0.5841068
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7112055, 0.7113945
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5656447, 0.5669668
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4347136, 0.4352905

Time for backsubstitution: 8.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3955291, upper bound: 0.3976667
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3956266, upper bound: 0.3975695
time: 3.57 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.17 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.17
Output dim: 7, lower bound: -0.3941637, upper bound: 0.3920048
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.17
Output dim: 7, lower bound: -0.3941637, upper bound: 0.3920048
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.17
Output dim: 7, lower bound: -0.3955291, upper bound: 0.3976667
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.17
Output dim: 7, lower bound: -0.3956266, upper bound: 0.3975695

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7267613, 0.7267101
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5886202, 0.5878801
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4784927, 0.4787849
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7316597, 0.7312167
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5537210, 0.5507853
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4364541, 0.4359962
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5841198, 0.5851952
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7112026, 0.7115440
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5666723, 0.5648899
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4352533, 0.4348851

Time for backsubstitution: 8.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2809

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 570

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3914401, upper bound: 0.3893398
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3915084, upper bound: 0.3892722
time: 3.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7268925, 0.7267362
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5889432, 0.5882621
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4786503, 0.4779789
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7317603, 0.7314517
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5537636, 0.5509386
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4361073, 0.4363503
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5839963, 0.5853057
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7113943, 0.7110133
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5669668, 0.5653504
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4352905, 0.4346765

Time for backsubstitution: 8.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 914

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3936382, upper bound: 0.3848803
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3870388, upper bound: 0.3914791
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7217021, 0.7222611
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5711751, 0.5729785
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4745783, 0.4741975
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7248788, 0.7240560
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5507557, 0.5537933
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4354763, 0.4358723
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5714759, 0.5699890
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7112055, 0.7113943
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5459161, 0.5477104
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4298232, 0.4312295

Time for backsubstitution: 8.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1992

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1151

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3946667, upper bound: 0.3939154
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3915283, upper bound: 0.3967882
time: 3.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7220788, 0.7218843
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5726199, 0.5715337
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4736837, 0.4750921
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7235122, 0.7254226
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5508149, 0.5537342
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4357612, 0.4355874
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5711879, 0.5702769
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7112055, 0.7113943
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5463884, 0.5472381
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4306527, 0.4303993

Time for backsubstitution: 8.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2809

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2586

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3707709, upper bound: 0.3730482
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3707709, upper bound: 0.3730482
time: 3.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.66 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3914401, upper bound: 0.3893398
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3915084, upper bound: 0.3892722
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3936382, upper bound: 0.3848803
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3870388, upper bound: 0.3914791
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3946667, upper bound: 0.3939154
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3915283, upper bound: 0.3967882
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3707709, upper bound: 0.3730482
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.66
Output dim: 7, lower bound: -0.3707709, upper bound: 0.3730482

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7217238, 0.7220527
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5711801, 0.5718849
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4749666, 0.4744357
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7250793, 0.7232697
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5535269, 0.5506504
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4355803, 0.4354072
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5703311, 0.5710802
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7112017, 0.7115438
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5469172, 0.5456071
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4303477, 0.4308099

Time for backsubstitution: 8.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 914

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1236

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3914401, upper bound: 0.3872487
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3893492, upper bound: 0.3893405
time: 3.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7221043, 0.7216760
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5726249, 0.5704329
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4740721, 0.4752588
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7236176, 0.7246363
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5535860, 0.5505912
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4358654, 0.4351223
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5700049, 0.5713682
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7112021, 0.7115438
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5473893, 0.5450934
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4311781, 0.4299804

Time for backsubstitution: 8.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 962

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3910946, upper bound: 0.3891344
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3913709, upper bound: 0.3889947
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7025733, 0.7076387
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5889289, 0.5883501
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4744899, 0.4715028
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7094808, 0.7049220
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5547030, 0.5459795
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4241813, 0.4274108
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5836689, 0.5852801
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7091742, 0.7074347
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5576255, 0.5516617
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4274194, 0.4303948

Time for backsubstitution: 8.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2468

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3916771, upper bound: 0.3811345
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3899015, upper bound: 0.3830063
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7077909, 0.7024212
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5890317, 0.5882475
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4722289, 0.4737637
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7052550, 0.7091473
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5488045, 0.5518780
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4271502, 0.4244420
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5839703, 0.5849787
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7078152, 0.7087936
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5532925, 0.5559952
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4310027, 0.4268113

Time for backsubstitution: 8.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1236

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3870388, upper bound: 0.3893891
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3849486, upper bound: 0.3914791
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7121401, 0.7135262
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5568403, 0.5663874
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4743544, 0.4739901
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7233653, 0.7215807
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5170355, 0.5142326
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4380834, 0.4383097
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5531448, 0.5599468
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7079239, 0.7077746
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5288978, 0.5307486
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4278836, 0.4295982

Time for backsubstitution: 8.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 914

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3941275, upper bound: 0.3871480
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3878078, upper bound: 0.3934017
time: 3.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7129688, 0.7126993
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5636564, 0.5586435
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4743633, 0.4739736
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7224038, 0.7225425
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5111952, 0.5198307
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4379137, 0.4384794
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5614353, 0.5516579
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7075858, 0.7081547
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5279200, 0.5306921
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4282441, 0.4292899

Time for backsubstitution: 8.91 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2867

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3850128, upper bound: 0.3905560
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3850128, upper bound: 0.3905560
time: 3.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7248707, 0.7217376
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5748010, 0.5714078
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4735606, 0.4750856
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7271154, 0.7253437
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5507610, 0.5540259
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4391944, 0.4354124
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5709585, 0.5727291
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7110605, 0.7087421
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5463855, 0.5472374
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4318831, 0.4303133

Time for backsubstitution: 9.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1859

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1243

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3699742, upper bound: 0.3695119
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3672406, upper bound: 0.3722519
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7219319, 0.7218843
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5724945, 0.5715337
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4736774, 0.4750921
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7234335, 0.7254226
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5508149, 0.5536802
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4355862, 0.4355874
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5711879, 0.5700476
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7112055, 0.7112494
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5463884, 0.5472351
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4305669, 0.4303993

Time for backsubstitution: 9.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2082

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2468

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3685255, upper bound: 0.3679711
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3656831, upper bound: 0.3708086
time: 3.94 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 16.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3914401, upper bound: 0.3872487
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3893492, upper bound: 0.3893405
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3910946, upper bound: 0.3891344
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3913709, upper bound: 0.3889947
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3916771, upper bound: 0.3811345
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3899015, upper bound: 0.3830063
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3870388, upper bound: 0.3893891
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3849486, upper bound: 0.3914791
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3941275, upper bound: 0.3871480
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3878078, upper bound: 0.3934017
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3850128, upper bound: 0.3905560
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3850128, upper bound: 0.3905560
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3699742, upper bound: 0.3695119
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3672406, upper bound: 0.3722519
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3685255, upper bound: 0.3679711
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.99
Output dim: 7, lower bound: -0.3656831, upper bound: 0.3708086

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7211022, 0.7219959
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5675938, 0.5662167
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4726930, 0.4731002
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7252584, 0.7232609
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5523748, 0.5489720
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4349046, 0.4349068
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5711582, 0.5722542
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7118073, 0.7124000
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5521097, 0.5479323
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4227829, 0.4248314

Time for backsubstitution: 9.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2082

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3827060, upper bound: 0.3693193
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3727564, upper bound: 0.3789283
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7216671, 0.7214310
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5655119, 0.5682991
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4736313, 0.4721619
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7250702, 0.7234490
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5518486, 0.5494983
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4350799, 0.4347316
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5715051, 0.5719073
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7120576, 0.7121491
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5492420, 0.5507997
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4243692, 0.4232451

Time for backsubstitution: 8.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2468

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3850932, upper bound: 0.3854301
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3855520, upper bound: 0.3848807
time: 3.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7217062, 0.7212906
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5731947, 0.5719123
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4743404, 0.4745578
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7235906, 0.7245753
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5535338, 0.5508628
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4349158, 0.4336870
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5625627, 0.5634949
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7110672, 0.7114677
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5466286, 0.5445554
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4300095, 0.4289305

Time for backsubstitution: 8.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 892

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3870184, upper bound: 0.3852186
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3874573, upper bound: 0.3847660
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7217188, 0.7212456
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5742211, 0.5710030
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4733710, 0.4754951
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7235909, 0.7246091
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5538576, 0.5505390
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4344264, 0.4341726
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5621486, 0.5639260
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7111235, 0.7114091
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5468515, 0.5442166
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4301265, 0.4288118

Time for backsubstitution: 8.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1859

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2082

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3825849, upper bound: 0.3693610
time: 3.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3726440, upper bound: 0.3796835
time: 4.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.6989927, 0.7025934
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5727909, 0.5784416
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4736724, 0.4707501
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.6665151, 0.6675020
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5549896, 0.5439754
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4248891, 0.4279218
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5836318, 0.5852810
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.6991830, 0.6980793
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5319090, 0.5261245
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4236038, 0.4269444

Time for backsubstitution: 8.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1773

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3819459, upper bound: 0.3680639
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3788113, upper bound: 0.3709861
time: 4.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.6975279, 0.7040583
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5788431, 0.5723894
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4737372, 0.4706852
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.6722012, 0.6618165
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5526989, 0.5462662
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4246495, 0.4281615
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5836701, 0.5852425
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.6998196, 0.6974380
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5319653, 0.5260673
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4239687, 0.4265795

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2818

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3496238, upper bound: 0.3442508
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3503932, upper bound: 0.3429575
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7071698, 0.7023647
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5854449, 0.5825787
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4699554, 0.4724286
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7054348, 0.7091389
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5476525, 0.5501997
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4264747, 0.4239414
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5847979, 0.5861526
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7084217, 0.7096505
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5584848, 0.5583205
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4234378, 0.4208329

Time for backsubstitution: 8.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2460

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1459

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3850884, upper bound: 0.3873899
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3850607, upper bound: 0.3874010
time: 3.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7077348, 0.7017998
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5833631, 0.5846605
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4708937, 0.4714901
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7052469, 0.7093271
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5471261, 0.5507259
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4266500, 0.4237661
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5851448, 0.5858057
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7086725, 0.7093997
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5556176, 0.5611880
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4250242, 0.4192464

Time for backsubstitution: 8.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 570
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2586

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 32

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3800438, upper bound: 0.3876807
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3810582, upper bound: 0.3873902
time: 3.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.6877618, 0.6943671
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5569155, 0.5665587
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4701288, 0.4675035
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7010512, 0.6950414
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5186524, 0.5099511
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4261740, 0.4293702
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5528157, 0.5599191
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7056656, 0.7041948
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5203943, 0.5175283
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4199241, 0.4252220

Time for backsubstitution: 9.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2468

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1859

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3939347, upper bound: 0.3858757
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3929329, upper bound: 0.3869524
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.6929851, 0.6891478
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5570114, 0.5664629
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4678679, 0.4697074
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.6968260, 0.6991320
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5127540, 0.5158496
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4291431, 0.4264003
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5531070, 0.5596178
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7043438, 0.7055538
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5156775, 0.5218617
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4235077, 0.4216387

Time for backsubstitution: 9.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 892

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775441, upper bound: 0.3831027
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3775441, upper bound: 0.3831027
time: 3.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7128084, 0.7126732
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5633022, 0.5579077
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4742379, 0.4747255
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7220678, 0.7223072
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5109935, 0.5196717
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4378990, 0.4381179
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5614944, 0.5515550
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7073932, 0.7084923
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5274432, 0.5297549
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4281927, 0.4294473

Time for backsubstitution: 9.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2145

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3833582, upper bound: 0.3873365
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3818627, upper bound: 0.3889888
time: 4.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7129426, 0.7126993
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5636564, 0.5582894
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4743633, 0.4738481
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7221684, 0.7225425
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5110362, 0.5198307
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4375521, 0.4384794
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5613322, 0.5516579
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7075858, 0.7079616
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5279200, 0.5302154
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4282441, 0.4292386

Time for backsubstitution: 9.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2468

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2145

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3833582, upper bound: 0.3873365
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3818627, upper bound: 0.3889888
time: 3.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7255299, 0.7212478
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5637963, 0.5687442
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4752384, 0.4749705
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7223916, 0.7234256
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5272338, 0.5148206
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4364654, 0.4345913
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5475246, 0.5427521
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7109432, 0.7084970
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5424247, 0.5457278
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4321523, 0.4302987

Time for backsubstitution: 9.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 892

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1773

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3583031, upper bound: 0.3552729
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3560039, upper bound: 0.3579614
time: 3.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7243807, 0.7223970
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5721328, 0.5604031
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4734454, 0.4767635
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7251973, 0.7206199
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5115557, 0.5304987
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4383734, 0.4326833
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5409814, 0.5492948
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7108154, 0.7086244
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5448761, 0.5432761
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4318684, 0.4305825

Time for backsubstitution: 9.24 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2809

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1236

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3672406, upper bound: 0.3701168
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3646234, upper bound: 0.3722519
time: 3.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7183518, 0.7168388
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5563564, 0.5614424
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4728596, 0.4743392
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.6803284, 0.6880033
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5511026, 0.5516770
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4363366, 0.4360982
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5711505, 0.5700486
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7012100, 0.7018948
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5209434, 0.5218463
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4267510, 0.4269482

Time for backsubstitution: 8.80 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2460

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 914

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3679129, upper bound: 0.3574492
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3579943, upper bound: 0.3673710
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7168870, 0.7183036
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5624089, 0.5553954
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4729245, 0.4742744
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.6860137, 0.6823177
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5488119, 0.5539677
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4360970, 0.4363378
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5711889, 0.5700102
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7018509, 0.7012539
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5210006, 0.5217900
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4271159, 0.4265834

Time for backsubstitution: 8.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2867
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1236
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 962

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2867

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3597612, upper bound: 0.3645110
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3597612, upper bound: 0.3645110
time: 3.20 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.25 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3827060, upper bound: 0.3693193
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3727564, upper bound: 0.3789283
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3850932, upper bound: 0.3854301
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3855520, upper bound: 0.3848807
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3870184, upper bound: 0.3852186
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3874573, upper bound: 0.3847660
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3825849, upper bound: 0.3693610
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3726440, upper bound: 0.3796835
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3819459, upper bound: 0.3680639
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3788113, upper bound: 0.3709861
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3496238, upper bound: 0.3442508
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3503932, upper bound: 0.3429575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3850884, upper bound: 0.3873899
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3850607, upper bound: 0.3874010
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3800438, upper bound: 0.3876807
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3810582, upper bound: 0.3873902
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3939347, upper bound: 0.3858757
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3929329, upper bound: 0.3869524
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3775441, upper bound: 0.3831027
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3775441, upper bound: 0.3831027
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3833582, upper bound: 0.3873365
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3818627, upper bound: 0.3889888
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3833582, upper bound: 0.3873365
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3818627, upper bound: 0.3889888
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3583031, upper bound: 0.3552729
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3560039, upper bound: 0.3579614
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3672406, upper bound: 0.3701168
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3646234, upper bound: 0.3722519
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3679129, upper bound: 0.3574492
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3579943, upper bound: 0.3673710
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3597612, upper bound: 0.3645110
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.25
Output dim: 7, lower bound: -0.3597612, upper bound: 0.3645110

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7174149, 0.7229408
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5610280, 0.5653484
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4727248, 0.4725938
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7201304, 0.7123904
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5485382, 0.5449359
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4337867, 0.4341055
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5670671, 0.5711744
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7145610, 0.7106237
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5517714, 0.5479442
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4208167, 0.4261149

Time for backsubstitution: 8.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1859

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3825132, upper bound: 0.3681325
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3815268, upper bound: 0.3691306
time: 3.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7211022, 0.7183087
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5675938, 0.5596509
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4721866, 0.4731002
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7252584, 0.7181330
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5483387, 0.5489720
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4349046, 0.4337891
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5700784, 0.5722542
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7100306, 0.7124000
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5521097, 0.5475941
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4227829, 0.4228652

Time for backsubstitution: 8.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 32
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2460

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3720735, upper bound: 0.3786857
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3725133, upper bound: 0.3783566
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7206011, 0.7209330
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5625575, 0.5677629
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4701209, 0.4676040
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7246096, 0.7225540
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5504618, 0.5481883
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4306293, 0.4301740
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5708199, 0.5722718
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7100892, 0.7093363
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5477951, 0.5461848
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4217885, 0.4212511

Time for backsubstitution: 8.90 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1859
type: DSZ, layer: 3, pos: 962
type: DSZ, layer: 3, pos: 2809
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1459
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1992
type: DSZ, layer: 3, pos: 2468
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2803
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 914
type: DSZ, layer: 3, pos: 892
type: DSZ, layer: 3, pos: 2586
type: DSZ, layer: 3, pos: 1151

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1859

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3849004, upper bound: 0.3842598
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3839200, upper bound: 0.3852373
time: 3.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.1927280, -13.9735336, -15.1927280, -13.9735336, -0.7216671, 0.7203649
1: -10.1100082, -9.0974617, -10.1100082, -9.0974617, -0.5655119, 0.5653443
2: -4.2049417, -3.3283174, -4.2049417, -3.3283174, -0.4690734, 0.4721619
3: -3.1393237, -1.9232740, -3.1393237, -1.9232740, -0.7250702, 0.7229884
4: -3.6613157, -2.8445070, -3.6613157, -2.8445070, -0.5505385, 0.5494983
5: -9.2376575, -8.4610548, -9.2376575, -8.4610548, -0.4350799, 0.4302810
6: -14.7872343, -13.7731915, -14.7872343, -13.7731915, -0.5715051, 0.5712222
7: 3.0774527, 3.8798275, 3.0774527, 3.8798275, -0.7120576, 0.7101808
8: -6.7023869, -5.8010044, -6.7023869, -5.8010044, -0.5446274, 0.5507997
9: -1.3182242, -0.6008139, -1.3182242, -0.6008139, -0.4243692, 0.4206643

Time for backsubstitution: 8.92 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.57 + 548.72 = 606.29 seconds
