## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0012634299999999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0028880, 0.0028880)
1: (-0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0007196, 0.0007196)
2: (0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0038136, 0.0038136)
3: (-0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0017358, 0.0017358)
4: (0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007381, 0.0007381)
5: (0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047965, 0.0047965)
6: (-0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0012174, 0.0012174)
7: (-0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0031498, 0.0031498)
8: (-0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016564, 0.0016564)
9: (0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0019207, 0.0019207)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.98 + 1.99 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0018048, upper bound: 0.0018049

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016488, upper bound: 0.0016703
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016703, upper bound: 0.0016488
time: 1.31 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.56 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.56
Output dim: 0, lower bound: -0.0016488, upper bound: 0.0016703
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.56
Output dim: 0, lower bound: -0.0016703, upper bound: 0.0016488

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0027900, 0.0028545
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006952, 0.0007113
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0037693, 0.0036841
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0016768, 0.0017156
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007295, 0.0007130
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047408, 0.0046336
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0011761, 0.0012033
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0030428, 0.0031132
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016002, 0.0016372
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018984, 0.0018555

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016346, upper bound: 0.0016484
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0016263, upper bound: 0.0016572
time: 1.29 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0028880, 0.0027900
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0007196, 0.0006952
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0036841, 0.0038136
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0017358, 0.0016768
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007130, 0.0007381
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0046336, 0.0047965
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0012174, 0.0011761
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0031498, 0.0030428
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016564, 0.0016002
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018555, 0.0019207

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011548, upper bound: 0.0011548
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011548, upper bound: 0.0011548
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 2.44 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0016346, upper bound: 0.0016484
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0016263, upper bound: 0.0016572
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0011548, upper bound: 0.0011548
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 2.44
Output dim: 0, lower bound: -0.0011548, upper bound: 0.0011548

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0027842, 0.0028461
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006937, 0.0007092
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0037583, 0.0036765
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0016734, 0.0017106
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007274, 0.0007116
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047269, 0.0046240
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0011736, 0.0011997
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0030365, 0.0031041
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0015969, 0.0016324
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018929, 0.0018517

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015076, upper bound: 0.0015728
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015541, upper bound: 0.0015234
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0027823, 0.0028487
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006933, 0.0007098
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0037616, 0.0036740
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0016722, 0.0017121
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007281, 0.0007111
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047311, 0.0046209
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0011728, 0.0012008
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0030345, 0.0031069
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0015958, 0.0016339
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018946, 0.0018504

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012519, upper bound: 0.0012590
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012519, upper bound: 0.0012590
time: 0.89 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 2.65 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0015076, upper bound: 0.0015728
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0015541, upper bound: 0.0015234
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0012519, upper bound: 0.0012590
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0012519, upper bound: 0.0012590

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0023220, 0.0024821
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005786, 0.0006185
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032776, 0.0030662
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0013956, 0.0014918
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006344, 0.0005935
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041223, 0.0038565
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0009788, 0.0010463
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0025325, 0.0027071
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013318, 0.0014236
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016508, 0.0015443

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014384, upper bound: 0.0014642
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013942, upper bound: 0.0014990
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024107, 0.0023840
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006007, 0.0005940
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0031480, 0.0031833
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014489, 0.0014328
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006093, 0.0006161
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0039594, 0.0040037
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010162, 0.0010049
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026292, 0.0026001
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013827, 0.0013673
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0015855, 0.0016033

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014423, upper bound: 0.0013924
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014423, upper bound: 0.0013936
time: 1.07 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0014384, upper bound: 0.0014642
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0013942, upper bound: 0.0014990
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0014423, upper bound: 0.0013924
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.00
Output dim: 0, lower bound: -0.0014423, upper bound: 0.0013936

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019004, 0.0020110
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004735, 0.0005011
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026555, 0.0025095
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011422, 0.0012087
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005140, 0.0004857
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033399, 0.0031563
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008011, 0.0008477
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020727, 0.0021933
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010900, 0.0011534
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013375, 0.0012639

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009681, upper bound: 0.0009720
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009681, upper bound: 0.0009720
time: 0.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018510, 0.0020387
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004612, 0.0005080
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026921, 0.0024442
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011125, 0.0012253
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005211, 0.0004731
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033860, 0.0030741
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007802, 0.0008594
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020187, 0.0022235
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010616, 0.0011693
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013559, 0.0012310

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013472, upper bound: 0.0014574
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013472, upper bound: 0.0014703
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024031, 0.0023729
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005988, 0.0005913
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0031334, 0.0031733
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014443, 0.0014262
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006065, 0.0006142
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0039410, 0.0039912
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010130, 0.0010003
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026209, 0.0025880
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013783, 0.0013610
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0015782, 0.0015982

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013640, upper bound: 0.0012836
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013328, upper bound: 0.0013118
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0023996, 0.0023840
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005979, 0.0005940
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0031480, 0.0031687
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014422, 0.0014328
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006093, 0.0006133
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0039594, 0.0039854
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010115, 0.0010049
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026171, 0.0026001
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013763, 0.0013673
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0015855, 0.0015959

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014300, upper bound: 0.0013722
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014224, upper bound: 0.0013815
time: 1.19 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0009681, upper bound: 0.0009720
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0009681, upper bound: 0.0009720
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0013472, upper bound: 0.0014574
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0013472, upper bound: 0.0014703
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0013640, upper bound: 0.0012836
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0013328, upper bound: 0.0013118
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0014300, upper bound: 0.0013722
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0014224, upper bound: 0.0013815

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017538, 0.0019283
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004370, 0.0004805
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025463, 0.0023159
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010541, 0.0011589
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004928, 0.0004482
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032025, 0.0029128
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007393, 0.0008128
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019128, 0.0021031
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010059, 0.0011060
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012824, 0.0011664

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012549, upper bound: 0.0013219
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012547, upper bound: 0.0013224
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017405, 0.0019386
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004337, 0.0004831
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025600, 0.0022983
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010461, 0.0011652
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004955, 0.0004448
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032198, 0.0028907
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007337, 0.0008172
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018983, 0.0021144
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009983, 0.0011119
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012893, 0.0011576

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009169, upper bound: 0.0009558
time: 0.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009169, upper bound: 0.0009558
time: 0.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019833, 0.0019033
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004942, 0.0004742
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025133, 0.0026189
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011920, 0.0011439
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004864, 0.0005069
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031610, 0.0032939
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008360, 0.0008023
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021631, 0.0020758
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011375, 0.0010916
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012658, 0.0013190

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013347, upper bound: 0.0012400
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013269, upper bound: 0.0012568
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019335, 0.0019337
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004818, 0.0004818
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025534, 0.0025531
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011621, 0.0011622
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004942, 0.0004942
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032116, 0.0032112
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008150, 0.0008151
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021087, 0.0021090
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011090, 0.0011091
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012861, 0.0012859

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013204, upper bound: 0.0012919
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0012993
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0023152, 0.0022931
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005769, 0.0005714
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0030280, 0.0030572
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0013915, 0.0013782
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005861, 0.0005917
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0038084, 0.0038452
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0009760, 0.0009666
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0025251, 0.0025009
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013279, 0.0013152
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0015251, 0.0015398

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012649
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013204, upper bound: 0.0012937
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0023090, 0.0022934
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005753, 0.0005714
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0030284, 0.0030490
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0013878, 0.0013784
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005861, 0.0005901
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0038089, 0.0038348
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0009733, 0.0009667
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0025183, 0.0025013
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013243, 0.0013154
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0015253, 0.0015356

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013449, upper bound: 0.0012716
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0013034
time: 1.13 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0012549, upper bound: 0.0013219
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0012547, upper bound: 0.0013224
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0009169, upper bound: 0.0009558
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0009169, upper bound: 0.0009558
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013347, upper bound: 0.0012400
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013269, upper bound: 0.0012568
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013204, upper bound: 0.0012919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0012993
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012649
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013204, upper bound: 0.0012937
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013449, upper bound: 0.0012716
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.23
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0013034

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017488, 0.0019191
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004358, 0.0004782
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025342, 0.0023093
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010511, 0.0011534
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004905, 0.0004470
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031873, 0.0029045
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007372, 0.0008090
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019074, 0.0020931
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010031, 0.0011007
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012763, 0.0011631

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0013067
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012310, upper bound: 0.0013094
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017447, 0.0019283
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004347, 0.0004805
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025463, 0.0023038
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010486, 0.0011589
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004928, 0.0004459
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032025, 0.0028976
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007354, 0.0008128
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019028, 0.0021031
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010007, 0.0011060
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012824, 0.0011603

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012426, upper bound: 0.0013071
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012309, upper bound: 0.0013100
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018825, 0.0017933
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004691, 0.0004468
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023680, 0.0024859
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011315, 0.0010778
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004583, 0.0004811
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029784, 0.0031266
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007936, 0.0007559
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020532, 0.0019558
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010798, 0.0010286
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011927, 0.0012520

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013226, upper bound: 0.0012243
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013155, upper bound: 0.0012276
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018733, 0.0018062
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004668, 0.0004501
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023850, 0.0024737
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011259, 0.0010856
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004616, 0.0004788
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029998, 0.0031112
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007897, 0.0007614
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020431, 0.0019699
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010745, 0.0010359
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012012, 0.0012459

Time for backsubstitution: 1.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 201

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013147, upper bound: 0.0012372
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013095, upper bound: 0.0012446
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018358, 0.0018296
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004574, 0.0004559
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024160, 0.0024241
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011034, 0.0010997
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004676, 0.0004692
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030387, 0.0030489
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007738, 0.0007713
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020022, 0.0019955
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010529, 0.0010494
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012168, 0.0012209

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012538
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012632
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018294, 0.0018334
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004558, 0.0004568
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024210, 0.0024157
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010995, 0.0011020
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004686, 0.0004675
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030450, 0.0030383
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007711, 0.0007729
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019952, 0.0019996
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010493, 0.0010516
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012194, 0.0012167

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012600
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012743, upper bound: 0.0012708
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018808, 0.0018088
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004686, 0.0004507
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023884, 0.0024836
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011304, 0.0010871
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004623, 0.0004807
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030040, 0.0031237
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007928, 0.0007625
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020513, 0.0019727
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010787, 0.0010374
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012029, 0.0012509

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013217, upper bound: 0.0012243
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0012374
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018322, 0.0018392
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004565, 0.0004583
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024286, 0.0024194
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011012, 0.0011054
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004701, 0.0004683
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030546, 0.0030429
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007723, 0.0007753
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019982, 0.0020059
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010509, 0.0010549
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012232, 0.0012185

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012551
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012651
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018744, 0.0018091
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004670, 0.0004508
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023888, 0.0024751
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011266, 0.0010873
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004624, 0.0004791
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030045, 0.0031130
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007901, 0.0007626
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020443, 0.0019730
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010751, 0.0010376
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012031, 0.0012466

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013151, upper bound: 0.0012277
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013093, upper bound: 0.0012450
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018259, 0.0018430
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004550, 0.0004592
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024337, 0.0024111
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010974, 0.0011077
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004710, 0.0004667
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030609, 0.0030325
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007697, 0.0007769
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019914, 0.0020101
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010473, 0.0010571
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012257, 0.0012143

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012629
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012755
time: 1.23 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0013067
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012310, upper bound: 0.0013094
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012426, upper bound: 0.0013071
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012309, upper bound: 0.0013100
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013226, upper bound: 0.0012243
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013155, upper bound: 0.0012276
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013147, upper bound: 0.0012372
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013095, upper bound: 0.0012446
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012538
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012632
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012600
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012743, upper bound: 0.0012708
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013217, upper bound: 0.0012243
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0012374
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012551
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012651
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013151, upper bound: 0.0012277
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0013093, upper bound: 0.0012450
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012629
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.16
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012755

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0016709, 0.0018279
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004163, 0.0004555
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024138, 0.0022064
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010043, 0.0010986
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004672, 0.0004270
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030359, 0.0027751
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007043, 0.0007705
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018224, 0.0019936
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009584, 0.0010484
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012157, 0.0011113

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012022, upper bound: 0.0012526
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011646, upper bound: 0.0012622
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0016577, 0.0018311
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004130, 0.0004563
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024179, 0.0021889
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0009963, 0.0011005
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004680, 0.0004237
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030411, 0.0027531
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0006988, 0.0007719
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018079, 0.0019970
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009508, 0.0010502
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012178, 0.0011025

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009222, upper bound: 0.0009831
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009222, upper bound: 0.0009831
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0016661, 0.0018375
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004151, 0.0004579
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024264, 0.0022000
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010014, 0.0011044
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004696, 0.0004258
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030518, 0.0027671
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007023, 0.0007746
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018171, 0.0020041
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009556, 0.0010539
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012221, 0.0011081

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 54

Time for candidate selection: 3.63 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010468, upper bound: 0.0011700
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010813, upper bound: 0.0011700
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0016535, 0.0018406
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004120, 0.0004586
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024306, 0.0021834
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0009938, 0.0011063
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004704, 0.0004226
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030570, 0.0027462
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0006970, 0.0007759
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018034, 0.0020075
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009484, 0.0010557
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012242, 0.0010997

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 139

Time for candidate selection: 3.67 seconds

### Candidate
type: DSZ, layer: 3, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011519, upper bound: 0.0012140
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011519, upper bound: 0.0012140
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017995, 0.0017021
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004484, 0.0004241
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022476, 0.0023762
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010815, 0.0010230
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004350, 0.0004599
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028269, 0.0029886
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007585, 0.0007175
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019626, 0.0018564
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010321, 0.0009763
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011320, 0.0011968

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010292, upper bound: 0.0009236
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010324, upper bound: 0.0008976
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017914, 0.0017024
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004464, 0.0004242
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022480, 0.0023655
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010767, 0.0010232
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004351, 0.0004578
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028274, 0.0029751
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007551, 0.0007176
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019537, 0.0018567
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010274, 0.0009764
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011322, 0.0011914

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 152

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010727
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010718
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017886, 0.0017150
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004457, 0.0004273
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022646, 0.0023618
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010750, 0.0010308
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004383, 0.0004571
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028483, 0.0029705
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007539, 0.0007229
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019507, 0.0018704
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010259, 0.0009836
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011406, 0.0011895

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012578, upper bound: 0.0011609
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012585, upper bound: 0.0011866
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017821, 0.0017172
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004441, 0.0004279
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022675, 0.0023533
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010711, 0.0010321
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004389, 0.0004555
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028519, 0.0029598
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007512, 0.0007238
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019437, 0.0018728
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010221, 0.0009849
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011420, 0.0011852

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012059, upper bound: 0.0012368
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012981, upper bound: 0.0012282
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017514, 0.0017325
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004364, 0.0004317
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022878, 0.0023127
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010526, 0.0010413
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004428, 0.0004476
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028775, 0.0029087
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007383, 0.0007303
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019101, 0.0018896
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010045, 0.0009937
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011523, 0.0011648

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 241

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012085, upper bound: 0.0011681
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012085, upper bound: 0.0011681
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017387, 0.0017448
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004332, 0.0004347
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023039, 0.0022959
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010450, 0.0010486
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004459, 0.0004444
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028977, 0.0028877
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007329, 0.0007355
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018963, 0.0019029
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009972, 0.0010007
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011604, 0.0011564

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 152

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011500, upper bound: 0.0011319
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011511, upper bound: 0.0011318
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017432, 0.0017364
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004344, 0.0004327
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022929, 0.0023019
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010477, 0.0010436
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004438, 0.0004455
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028838, 0.0028952
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007348, 0.0007319
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019012, 0.0018938
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009998, 0.0009959
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011548, 0.0011594

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012059, upper bound: 0.0012518
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012099, upper bound: 0.0012520
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017323, 0.0017506
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004316, 0.0004362
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023116, 0.0022875
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010412, 0.0010522
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004474, 0.0004427
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029074, 0.0028771
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007302, 0.0007379
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018893, 0.0019093
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009936, 0.0010041
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011643, 0.0011521

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 248

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012059, upper bound: 0.0012629
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012662, upper bound: 0.0012627
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017946, 0.0017117
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004472, 0.0004265
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022603, 0.0023698
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010786, 0.0010288
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004375, 0.0004587
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028428, 0.0029805
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007565, 0.0007215
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019573, 0.0018669
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010293, 0.0009818
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011384, 0.0011935

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 139

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 136

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 152

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010292, upper bound: 0.0009236
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010324, upper bound: 0.0008976
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017837, 0.0017246
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004445, 0.0004297
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022773, 0.0023554
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010721, 0.0010365
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004408, 0.0004559
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028642, 0.0029624
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007519, 0.0007270
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019454, 0.0018809
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010231, 0.0009891
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011470, 0.0011863

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011821, upper bound: 0.0010564
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011821, upper bound: 0.0010509
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017477, 0.0017421
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004355, 0.0004341
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023005, 0.0023078
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010504, 0.0010471
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004453, 0.0004467
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028934, 0.0029027
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007367, 0.0007344
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019061, 0.0019000
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010024, 0.0009992
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011586, 0.0011623

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 136

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010051, upper bound: 0.0009476
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010069, upper bound: 0.0009185
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017351, 0.0017543
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004323, 0.0004371
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023166, 0.0022912
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010428, 0.0010544
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004484, 0.0004435
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029137, 0.0028817
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007314, 0.0007395
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018924, 0.0019134
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009952, 0.0010062
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011668, 0.0011540

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012400, upper bound: 0.0011964
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012399, upper bound: 0.0012181
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017869, 0.0017120
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004452, 0.0004266
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022607, 0.0023596
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010740, 0.0010290
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004375, 0.0004567
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028433, 0.0029677
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007532, 0.0007217
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019489, 0.0018672
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010249, 0.0009819
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011386, 0.0011884

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 152

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010727
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010718
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017773, 0.0017267
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004429, 0.0004303
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022802, 0.0023469
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010682, 0.0010378
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004413, 0.0004542
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028678, 0.0029518
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007492, 0.0007279
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019384, 0.0018833
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010194, 0.0009904
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011484, 0.0011820

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012520, upper bound: 0.0011955
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012523, upper bound: 0.0011955
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017400, 0.0017460
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004336, 0.0004350
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023055, 0.0022977
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010458, 0.0010494
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004462, 0.0004447
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028997, 0.0028898
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007335, 0.0007360
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018977, 0.0019042
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009980, 0.0010014
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011612, 0.0011572

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 54

Time for candidate selection: 1.83 seconds

### Candidate
type: DSZ, layer: 3, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011995, upper bound: 0.0011777
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011995, upper bound: 0.0011777
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017288, 0.0017602
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004308, 0.0004386
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023243, 0.0022829
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010391, 0.0010579
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004499, 0.0004418
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029234, 0.0028713
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007288, 0.0007420
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018855, 0.0019197
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009916, 0.0010096
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011706, 0.0011498

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 83

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 248
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 54

Time for candidate selection: 1.88 seconds

### Candidate
type: DSZ, layer: 3, pos: 165

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012216, upper bound: 0.0012254
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0012216, upper bound: 0.0012255
time: 1.11 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 5.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012022, upper bound: 0.0012526
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011646, upper bound: 0.0012622
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0009222, upper bound: 0.0009831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0009222, upper bound: 0.0009831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010468, upper bound: 0.0011700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010813, upper bound: 0.0011700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011519, upper bound: 0.0012140
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011519, upper bound: 0.0012140
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010292, upper bound: 0.0009236
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010324, upper bound: 0.0008976
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010727
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010718
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012578, upper bound: 0.0011609
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012585, upper bound: 0.0011866
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012059, upper bound: 0.0012368
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012981, upper bound: 0.0012282
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012085, upper bound: 0.0011681
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012085, upper bound: 0.0011681
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011500, upper bound: 0.0011319
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011511, upper bound: 0.0011318
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012059, upper bound: 0.0012518
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012099, upper bound: 0.0012520
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012059, upper bound: 0.0012629
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012662, upper bound: 0.0012627
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010292, upper bound: 0.0009236
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010324, upper bound: 0.0008976
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011821, upper bound: 0.0010564
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011821, upper bound: 0.0010509
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010051, upper bound: 0.0009476
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0010069, upper bound: 0.0009185
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012400, upper bound: 0.0011964
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012399, upper bound: 0.0012181
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010727
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011803, upper bound: 0.0010718
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012520, upper bound: 0.0011955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012523, upper bound: 0.0011955
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011995, upper bound: 0.0011777
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0011995, upper bound: 0.0011777
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012216, upper bound: 0.0012254
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 5.17
Output dim: 0, lower bound: -0.0012216, upper bound: 0.0012255

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017725, 0.0017093
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004417, 0.0004259
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022571, 0.0023406
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010653, 0.0010273
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004368, 0.0004530
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028388, 0.0029438
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007472, 0.0007205
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019332, 0.0018642
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010166, 0.0009804
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011368, 0.0011788

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 241

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011677, upper bound: 0.0010739
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011677, upper bound: 0.0010729
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017236, 0.0017427
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004295, 0.0004342
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023012, 0.0022760
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010359, 0.0010474
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004454, 0.0004405
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028943, 0.0028626
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007266, 0.0007346
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018798, 0.0019007
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009886, 0.0009995
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011590, 0.0011463

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 67

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 139

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009434
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009434
time: 0.99 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.69 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.69
Output dim: 0, lower bound: -0.0011677, upper bound: 0.0010739
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.69
Output dim: 0, lower bound: -0.0011677, upper bound: 0.0010729
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.69
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009434
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.69
Output dim: 0, lower bound: -0.0009562, upper bound: 0.0009434

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.98 + 206.82 = 209.80 seconds
