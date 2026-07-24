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
execution time: IAR + RelationalAnalysis = 1.41 + 1.99 = 3.40 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0018048, upper bound: 0.0018049

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 56
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 56

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017378, upper bound: 0.0017313
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017378
time: 1.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.73 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.73
Output dim: 0, lower bound: -0.0017378, upper bound: 0.0017313
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.73
Output dim: 0, lower bound: -0.0017313, upper bound: 0.0017378

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0028833, 0.0028793
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0007184, 0.0007174
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0038021, 0.0038073
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0017329, 0.0017305
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007359, 0.0007369
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047820, 0.0047886
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0012154, 0.0012137
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0031446, 0.0031403
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016537, 0.0016514
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0019149, 0.0019176

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015175, upper bound: 0.0015228
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015228, upper bound: 0.0015175
time: 0.97 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0028793, 0.0028880
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0007174, 0.0007196
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0038136, 0.0038021
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0017305, 0.0017358
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007381, 0.0007359
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047965, 0.0047820
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0012137, 0.0012174
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0031403, 0.0031498
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016514, 0.0016564
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0019207, 0.0019149

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015175, upper bound: 0.0015228
time: 1.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0015228, upper bound: 0.0015175
time: 1.01 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.33 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 0, lower bound: -0.0015175, upper bound: 0.0015228
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 0, lower bound: -0.0015228, upper bound: 0.0015175
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 0, lower bound: -0.0015175, upper bound: 0.0015228
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.33
Output dim: 0, lower bound: -0.0015228, upper bound: 0.0015175

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0027852, 0.0028457
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006940, 0.0007091
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0037578, 0.0036778
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0016740, 0.0017104
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007273, 0.0007118
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047263, 0.0046258
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0011741, 0.0011996
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0030377, 0.0031037
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0015975, 0.0016322
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018926, 0.0018524

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0014132
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014068, upper bound: 0.0014428
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0028833, 0.0027812
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0007184, 0.0006930
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0036726, 0.0038073
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0017329, 0.0016716
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007108, 0.0007369
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0046192, 0.0047886
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0012154, 0.0011724
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0031446, 0.0030333
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016537, 0.0015952
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018497, 0.0019176

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014428, upper bound: 0.0014068
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014132, upper bound: 0.0014395
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0027812, 0.0028545
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006930, 0.0007113
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0037693, 0.0036726
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0016716, 0.0017156
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007295, 0.0007108
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0047408, 0.0046192
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0011724, 0.0012033
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0030333, 0.0031132
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0015952, 0.0016372
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018984, 0.0018497

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0014132
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014068, upper bound: 0.0014428
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0028793, 0.0027900
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0007174, 0.0006952
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0036841, 0.0038021
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0017305, 0.0016768
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0007130, 0.0007359
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0046336, 0.0047820
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0012137, 0.0011761
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0031403, 0.0030428
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0016514, 0.0016002
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0018555, 0.0019149

Time for backsubstitution: 1.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014428, upper bound: 0.0014068
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014132, upper bound: 0.0014395
time: 1.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0014132
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014068, upper bound: 0.0014428
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014428, upper bound: 0.0014068
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014132, upper bound: 0.0014395
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014395, upper bound: 0.0014132
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014068, upper bound: 0.0014428
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014428, upper bound: 0.0014068
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.45
Output dim: 0, lower bound: -0.0014132, upper bound: 0.0014395

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024966, 0.0025056
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006221, 0.0006243
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033086, 0.0032967
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015005, 0.0015059
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006404, 0.0006381
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041613, 0.0041463
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010524, 0.0010562
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027228, 0.0027327
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014319, 0.0014371
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016664, 0.0016604

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014245, upper bound: 0.0013914
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014151, upper bound: 0.0013990
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024451, 0.0025366
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006092, 0.0006320
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033495, 0.0032287
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014695, 0.0015245
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006483, 0.0006249
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0042128, 0.0040608
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010307, 0.0010693
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026667, 0.0027665
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014024, 0.0014549
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016870, 0.0016261

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013932, upper bound: 0.0014179
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013861, upper bound: 0.0014280
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0026095, 0.0024411
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006502, 0.0006083
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032234, 0.0034459
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015684, 0.0014672
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006239, 0.0006669
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0040542, 0.0043340
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0011000, 0.0010290
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0028461, 0.0026623
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014967, 0.0014001
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016235, 0.0017355

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014280, upper bound: 0.0013860
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014184, upper bound: 0.0013932
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025580, 0.0024906
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006374, 0.0006206
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032888, 0.0033779
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015375, 0.0014969
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006365, 0.0006538
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041364, 0.0042485
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010783, 0.0010499
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027899, 0.0027163
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014672, 0.0014285
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016564, 0.0017013

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013990, upper bound: 0.0014146
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0014244
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024906, 0.0025133
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006206, 0.0006262
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033187, 0.0032888
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014969, 0.0015105
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006423, 0.0006365
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041741, 0.0041364
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010499, 0.0010594
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027163, 0.0027411
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014285, 0.0014415
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016715, 0.0016564

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014244, upper bound: 0.0013914
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014146, upper bound: 0.0013990
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024411, 0.0025442
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006083, 0.0006340
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033596, 0.0032234
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014672, 0.0015292
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006502, 0.0006239
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0042255, 0.0040542
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010290, 0.0010725
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026623, 0.0027748
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014001, 0.0014593
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016921, 0.0016235

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013932, upper bound: 0.0014184
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013861, upper bound: 0.0014280
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0026035, 0.0024488
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006487, 0.0006102
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032335, 0.0034380
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015648, 0.0014718
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006258, 0.0006654
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0040670, 0.0043240
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010975, 0.0010322
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0028395, 0.0026707
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014933, 0.0014045
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016286, 0.0017315

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014280, upper bound: 0.0013861
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0014179, upper bound: 0.0013932
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025541, 0.0024982
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006364, 0.0006225
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032989, 0.0033726
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015351, 0.0015015
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006385, 0.0006528
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041491, 0.0042419
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010766, 0.0010531
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027856, 0.0027247
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014649, 0.0014329
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016615, 0.0016986

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013990, upper bound: 0.0014151
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0014245
time: 1.09 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014245, upper bound: 0.0013914
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014151, upper bound: 0.0013990
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013932, upper bound: 0.0014179
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013861, upper bound: 0.0014280
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014280, upper bound: 0.0013860
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014184, upper bound: 0.0013932
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013990, upper bound: 0.0014146
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0014244
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014244, upper bound: 0.0013914
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014146, upper bound: 0.0013990
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013932, upper bound: 0.0014184
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013861, upper bound: 0.0014280
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014280, upper bound: 0.0013861
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0014179, upper bound: 0.0013932
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013990, upper bound: 0.0014151
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.32
Output dim: 0, lower bound: -0.0013914, upper bound: 0.0014245

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024849, 0.0024914
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006192, 0.0006208
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032898, 0.0032813
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014935, 0.0014974
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006367, 0.0006351
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041377, 0.0041270
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010475, 0.0010502
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027101, 0.0027172
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014252, 0.0014289
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016569, 0.0016526

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013166, upper bound: 0.0013347
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013640, upper bound: 0.0012836
time: 1.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024824, 0.0024939
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006186, 0.0006214
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032932, 0.0032780
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014920, 0.0014989
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006374, 0.0006345
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041420, 0.0041229
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010464, 0.0010513
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027074, 0.0027200
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014238, 0.0014304
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016586, 0.0016510

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0013418
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013550, upper bound: 0.0012854
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024334, 0.0025218
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006063, 0.0006284
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033300, 0.0032133
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014625, 0.0015157
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006445, 0.0006219
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041883, 0.0040414
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010258, 0.0010630
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026540, 0.0027504
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013957, 0.0014464
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016772, 0.0016184

Time for backsubstitution: 1.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012821, upper bound: 0.0013610
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013328, upper bound: 0.0013118
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024315, 0.0025249
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006059, 0.0006291
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033341, 0.0032108
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014614, 0.0015175
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006453, 0.0006214
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041934, 0.0040383
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010250, 0.0010643
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026519, 0.0027538
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013946, 0.0014482
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016792, 0.0016171

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012808, upper bound: 0.0013698
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013259, upper bound: 0.0013150
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025975, 0.0024275
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006472, 0.0006049
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032055, 0.0034300
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015612, 0.0014590
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006204, 0.0006639
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0040317, 0.0043141
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010950, 0.0010233
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0028330, 0.0026476
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014898, 0.0013923
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016145, 0.0017275

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013178, upper bound: 0.0013259
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013703, upper bound: 0.0012805
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025951, 0.0024294
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006466, 0.0006053
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032080, 0.0034267
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015597, 0.0014602
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006209, 0.0006632
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0040349, 0.0043099
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010939, 0.0010241
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0028303, 0.0026496
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014884, 0.0013934
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016157, 0.0017259

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013157, upper bound: 0.0013328
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013615, upper bound: 0.0012820
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025460, 0.0024764
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006344, 0.0006171
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032701, 0.0033620
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015302, 0.0014884
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006329, 0.0006507
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041129, 0.0042285
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010732, 0.0010439
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027768, 0.0027009
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014603, 0.0014204
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016470, 0.0016933

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012856, upper bound: 0.0013540
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013418, upper bound: 0.0013100
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025441, 0.0024789
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006339, 0.0006177
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032734, 0.0033595
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015291, 0.0014899
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006336, 0.0006502
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041170, 0.0042254
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010724, 0.0010449
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027747, 0.0027036
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014592, 0.0014218
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016486, 0.0016920

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012840, upper bound: 0.0013634
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013347, upper bound: 0.0013133
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024789, 0.0024990
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006177, 0.0006227
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032999, 0.0032734
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014899, 0.0015020
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006387, 0.0006336
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041504, 0.0041170
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010449, 0.0010534
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027036, 0.0027255
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014218, 0.0014333
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016620, 0.0016486

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013347
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013634, upper bound: 0.0012840
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024764, 0.0025015
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006171, 0.0006233
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033032, 0.0032701
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014884, 0.0015035
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006393, 0.0006329
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041546, 0.0041129
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010439, 0.0010545
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027009, 0.0027283
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014204, 0.0014348
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016637, 0.0016470

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013100, upper bound: 0.0013418
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013540, upper bound: 0.0012856
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024294, 0.0025294
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006053, 0.0006303
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033401, 0.0032080
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014602, 0.0015203
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006465, 0.0006209
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0042009, 0.0040349
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010241, 0.0010662
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026496, 0.0027587
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013934, 0.0014508
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016822, 0.0016157

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012820, upper bound: 0.0013615
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013328, upper bound: 0.0013157
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0024275, 0.0025325
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006049, 0.0006310
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0033441, 0.0032055
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0014590, 0.0015221
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006473, 0.0006204
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0042061, 0.0040317
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010233, 0.0010675
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0026476, 0.0027621
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0013923, 0.0014525
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016843, 0.0016145

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012805, upper bound: 0.0013703
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013259, upper bound: 0.0013178
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025915, 0.0024351
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006457, 0.0006068
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032156, 0.0034221
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015576, 0.0014636
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006224, 0.0006623
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0040444, 0.0043041
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010924, 0.0010265
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0028264, 0.0026559
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014864, 0.0013967
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016195, 0.0017235

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013150, upper bound: 0.0013259
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013698, upper bound: 0.0012808
time: 1.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025891, 0.0024370
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006451, 0.0006072
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032181, 0.0034188
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015561, 0.0014647
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006229, 0.0006617
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0040475, 0.0043000
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010914, 0.0010273
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0028237, 0.0026579
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014850, 0.0013978
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016208, 0.0017219

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013118, upper bound: 0.0013328
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013610, upper bound: 0.0012821
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025421, 0.0024840
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006334, 0.0006190
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032801, 0.0033568
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015279, 0.0014930
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006349, 0.0006497
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041255, 0.0042219
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010716, 0.0010471
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027725, 0.0027092
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014580, 0.0014247
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016520, 0.0016906

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012854, upper bound: 0.0013550
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013418, upper bound: 0.0013143
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0025402, 0.0024865
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0006329, 0.0006196
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0032834, 0.0033543
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0015267, 0.0014945
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0006355, 0.0006492
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0041297, 0.0042188
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0010708, 0.0010482
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0027704, 0.0027119
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0014569, 0.0014262
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0016537, 0.0016894

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012836, upper bound: 0.0013640
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013347, upper bound: 0.0013166
time: 1.24 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.59 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013166, upper bound: 0.0013347
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013640, upper bound: 0.0012836
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013143, upper bound: 0.0013418
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013550, upper bound: 0.0012854
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012821, upper bound: 0.0013610
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013328, upper bound: 0.0013118
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012808, upper bound: 0.0013698
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013259, upper bound: 0.0013150
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013178, upper bound: 0.0013259
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013703, upper bound: 0.0012805
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013157, upper bound: 0.0013328
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013615, upper bound: 0.0012820
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012856, upper bound: 0.0013540
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013418, upper bound: 0.0013100
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012840, upper bound: 0.0013634
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013347, upper bound: 0.0013133
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013133, upper bound: 0.0013347
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013634, upper bound: 0.0012840
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013100, upper bound: 0.0013418
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013540, upper bound: 0.0012856
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012820, upper bound: 0.0013615
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013328, upper bound: 0.0013157
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012805, upper bound: 0.0013703
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013259, upper bound: 0.0013178
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013150, upper bound: 0.0013259
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013698, upper bound: 0.0012808
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013118, upper bound: 0.0013328
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013610, upper bound: 0.0012821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012854, upper bound: 0.0013550
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013418, upper bound: 0.0013143
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0012836, upper bound: 0.0013640
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.59
Output dim: 0, lower bound: -0.0013347, upper bound: 0.0013166

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018968, 0.0020014
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004726, 0.0004987
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026428, 0.0025047
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011400, 0.0012029
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005115, 0.0004848
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033240, 0.0031503
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007996, 0.0008437
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020687, 0.0021828
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010879, 0.0011479
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013311, 0.0012615

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013045, upper bound: 0.0013183
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012923, upper bound: 0.0013222
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013517, upper bound: 0.0012649
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013452, upper bound: 0.0012712
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018943, 0.0020065
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004720, 0.0005000
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026495, 0.0025014
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011385, 0.0012059
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005128, 0.0004841
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033324, 0.0031462
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007985, 0.0008458
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020660, 0.0021883
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010865, 0.0011508
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013344, 0.0012599

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013021, upper bound: 0.0013250
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012894, upper bound: 0.0013293
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019782, 0.0019058
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004929, 0.0004749
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025166, 0.0026122
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011889, 0.0011455
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004871, 0.0005056
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031653, 0.0032854
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008339, 0.0008034
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021575, 0.0020786
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011346, 0.0010931
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012675, 0.0013156

Time for backsubstitution: 1.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013427, upper bound: 0.0012664
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013365, upper bound: 0.0012731
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018453, 0.0020291
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004598, 0.0005056
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026794, 0.0024367
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011091, 0.0012195
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005186, 0.0004716
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033700, 0.0030647
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007779, 0.0008553
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020126, 0.0022130
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010584, 0.0011638
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013495, 0.0012273

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012576, upper bound: 0.0013450
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012590, upper bound: 0.0013485
time: 1.24 seconds

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

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

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
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018434, 0.0020345
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004593, 0.0005070
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026866, 0.0024342
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011079, 0.0012228
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005200, 0.0004711
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033790, 0.0030616
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007771, 0.0008576
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020105, 0.0022190
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010573, 0.0011669
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013531, 0.0012260

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012685, upper bound: 0.0013528
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012576, upper bound: 0.0013574
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019288, 0.0019368
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004806, 0.0004826
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025575, 0.0025470
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011593, 0.0011641
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004950, 0.0004930
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032167, 0.0032034
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008131, 0.0008164
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021036, 0.0021124
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011063, 0.0011109
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012881, 0.0012828

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013134, upper bound: 0.0012935
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013059, upper bound: 0.0013028
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019880, 0.0019253
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004954, 0.0004797
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025424, 0.0026252
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011949, 0.0011572
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004921, 0.0005081
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031977, 0.0033018
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008380, 0.0008116
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021682, 0.0020999
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011403, 0.0011043
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012805, 0.0013222

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013057, upper bound: 0.0013059
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012963, upper bound: 0.0013134
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020745, 0.0018395
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005169, 0.0004583
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024290, 0.0027394
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012468, 0.0011056
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004701, 0.0005302
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030550, 0.0034454
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008745, 0.0007754
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0022626, 0.0020062
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011899, 0.0010550
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012234, 0.0013797

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013579, upper bound: 0.0012574
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013530, upper bound: 0.0012681
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019856, 0.0019300
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004947, 0.0004809
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025485, 0.0026219
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011934, 0.0011600
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004933, 0.0005075
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032054, 0.0032977
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008370, 0.0008136
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021655, 0.0021049
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011388, 0.0011070
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012836, 0.0013205

Time for backsubstitution: 1.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013034, upper bound: 0.0013123
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012937, upper bound: 0.0013204
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020694, 0.0018413
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005156, 0.0004588
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024315, 0.0027326
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012438, 0.0011067
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004706, 0.0005289
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030581, 0.0034369
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008723, 0.0007762
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0022570, 0.0020082
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011869, 0.0010561
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012246, 0.0013763

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0012588
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013452, upper bound: 0.0012697
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019365, 0.0019732
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004825, 0.0004917
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026056, 0.0025572
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011639, 0.0011860
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005043, 0.0004949
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032771, 0.0032163
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008163, 0.0008318
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021121, 0.0021521
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011107, 0.0011317
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013123, 0.0012879

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012734, upper bound: 0.0013360
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012664, upper bound: 0.0013417
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020247, 0.0018883
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005045, 0.0004705
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024935, 0.0026736
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012169, 0.0011349
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004826, 0.0005175
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031362, 0.0033627
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008535, 0.0007960
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0022082, 0.0020595
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011613, 0.0010831
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012559, 0.0013466

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013293, upper bound: 0.0012870
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0012977
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019346, 0.0019785
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004821, 0.0004930
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026126, 0.0025547
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011628, 0.0011891
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005057, 0.0004944
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032859, 0.0032131
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008155, 0.0008340
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021100, 0.0021578
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011096, 0.0011348
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013158, 0.0012867

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012716, upper bound: 0.0013449
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013511
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020200, 0.0018908
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005033, 0.0004711
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024968, 0.0026674
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012141, 0.0011364
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004832, 0.0005163
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031403, 0.0033549
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008515, 0.0007970
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0022031, 0.0020622
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011586, 0.0010845
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012575, 0.0013435

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013222, upper bound: 0.0012895
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013183, upper bound: 0.0013011
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018908, 0.0020110
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004711, 0.0005011
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026555, 0.0024968
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011364, 0.0012087
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005140, 0.0004832
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033399, 0.0031403
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007970, 0.0008477
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020622, 0.0021933
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010845, 0.0011534
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013375, 0.0012575

Time for backsubstitution: 1.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013011, upper bound: 0.0013183
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012895, upper bound: 0.0013222
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019785, 0.0019129
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004930, 0.0004766
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025260, 0.0026126
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011891, 0.0011497
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004889, 0.0005057
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031770, 0.0032859
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008340, 0.0008064
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021578, 0.0020863
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011348, 0.0010972
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012722, 0.0013158

Time for backsubstitution: 1.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012649
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013449, upper bound: 0.0012716
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018883, 0.0020161
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004705, 0.0005024
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026622, 0.0024935
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011349, 0.0012117
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005153, 0.0004826
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033484, 0.0031362
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007960, 0.0008499
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020595, 0.0021988
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010831, 0.0011563
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013408, 0.0012559

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012977, upper bound: 0.0013250
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012870, upper bound: 0.0013293
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019732, 0.0019155
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004917, 0.0004773
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025293, 0.0026056
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011860, 0.0011512
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004895, 0.0005043
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031812, 0.0032771
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008318, 0.0008074
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021521, 0.0020891
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011317, 0.0010986
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012739, 0.0013123

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013417, upper bound: 0.0012664
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013360, upper bound: 0.0012734
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018413, 0.0020387
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004588, 0.0005080
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026921, 0.0024315
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011067, 0.0012253
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005211, 0.0004706
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033860, 0.0030581
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007762, 0.0008594
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020082, 0.0022235
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010561, 0.0011693
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013559, 0.0012246

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0013453
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012588, upper bound: 0.0013492
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019300, 0.0019433
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004809, 0.0004842
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025662, 0.0025485
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011600, 0.0011680
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004967, 0.0004933
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032276, 0.0032054
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008136, 0.0008192
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021049, 0.0021195
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011070, 0.0011146
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012925, 0.0012836

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013204, upper bound: 0.0012937
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0013034
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018395, 0.0020442
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004583, 0.0005094
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026993, 0.0024290
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011056, 0.0012286
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005224, 0.0004701
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033950, 0.0030550
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007754, 0.0008617
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020062, 0.0022295
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010550, 0.0011725
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013595, 0.0012234

Time for backsubstitution: 1.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012681, upper bound: 0.0013530
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0013579
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019253, 0.0019464
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004797, 0.0004850
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025702, 0.0025424
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011572, 0.0011699
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004975, 0.0004921
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032327, 0.0031977
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008116, 0.0008205
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020999, 0.0021229
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011043, 0.0011164
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012945, 0.0012805

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0012963
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013059, upper bound: 0.0013057
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019820, 0.0019350
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004939, 0.0004821
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025551, 0.0026173
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011913, 0.0011630
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004945, 0.0005066
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032137, 0.0032918
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008355, 0.0008157
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021617, 0.0021104
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011368, 0.0011098
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012869, 0.0013182

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013028, upper bound: 0.0013059
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012935, upper bound: 0.0013134
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020697, 0.0018491
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005157, 0.0004607
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024417, 0.0027330
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012440, 0.0011114
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004726, 0.0005290
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030710, 0.0034374
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008725, 0.0007795
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0022573, 0.0020167
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011871, 0.0010606
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012298, 0.0013765

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013574, upper bound: 0.0012576
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013528, upper bound: 0.0012685
time: 1.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019796, 0.0019396
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004933, 0.0004833
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025612, 0.0026140
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011898, 0.0011658
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004957, 0.0005059
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032214, 0.0032877
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008345, 0.0008176
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021590, 0.0021154
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011354, 0.0011125
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012900, 0.0013165

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012993, upper bound: 0.0013123
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012919, upper bound: 0.0013204
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020644, 0.0018510
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005144, 0.0004612
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024442, 0.0027261
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012408, 0.0011125
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004731, 0.0005276
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030741, 0.0034287
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008702, 0.0007802
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0022516, 0.0020187
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011841, 0.0010616
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012310, 0.0013730

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013485, upper bound: 0.0012590
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013450, upper bound: 0.0012698
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019326, 0.0019828
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004815, 0.0004941
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026183, 0.0025519
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011615, 0.0011917
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005068, 0.0004939
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032931, 0.0032097
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008146, 0.0008358
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021077, 0.0021626
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011084, 0.0011373
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013187, 0.0012853

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012731, upper bound: 0.0013365
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012663, upper bound: 0.0013427
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020212, 0.0018980
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005036, 0.0004729
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025062, 0.0026690
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012148, 0.0011407
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004851, 0.0005166
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031522, 0.0033569
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008520, 0.0008001
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0022044, 0.0020700
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011593, 0.0010886
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012623, 0.0013443

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013293, upper bound: 0.0012894
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0013021
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019307, 0.0019881
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004811, 0.0004954
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0026253, 0.0025494
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011604, 0.0011949
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0005081, 0.0004934
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0033019, 0.0032065
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008139, 0.0008381
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021057, 0.0021683
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011074, 0.0011403
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0013222, 0.0012840

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012712, upper bound: 0.0013452
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013517
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0020166, 0.0019004
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0005025, 0.0004735
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025095, 0.0026629
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0012120, 0.0011422
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004857, 0.0005154
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031563, 0.0033492
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008501, 0.0008011
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021994, 0.0020727
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011566, 0.0010900
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012639, 0.0013412

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 201

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013222, upper bound: 0.0012923
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013183, upper bound: 0.0013045
time: 1.16 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013045, upper bound: 0.0013183
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012923, upper bound: 0.0013222
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013517, upper bound: 0.0012649
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013452, upper bound: 0.0012712
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013021, upper bound: 0.0013250
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012894, upper bound: 0.0013293
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013427, upper bound: 0.0012664
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013365, upper bound: 0.0012731
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012576, upper bound: 0.0013450
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012590, upper bound: 0.0013485
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013204, upper bound: 0.0012919
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0012993
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012685, upper bound: 0.0013528
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012576, upper bound: 0.0013574
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013134, upper bound: 0.0012935
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013059, upper bound: 0.0013028
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013057, upper bound: 0.0013059
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012963, upper bound: 0.0013134
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013579, upper bound: 0.0012574
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013530, upper bound: 0.0012681
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013034, upper bound: 0.0013123
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012937, upper bound: 0.0013204
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013492, upper bound: 0.0012588
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013452, upper bound: 0.0012697
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012734, upper bound: 0.0013360
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012664, upper bound: 0.0013417
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013293, upper bound: 0.0012870
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0012977
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012716, upper bound: 0.0013449
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013511
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013222, upper bound: 0.0012895
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013183, upper bound: 0.0013011
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013011, upper bound: 0.0013183
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012895, upper bound: 0.0013222
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013511, upper bound: 0.0012649
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013449, upper bound: 0.0012716
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012977, upper bound: 0.0013250
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012870, upper bound: 0.0013293
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013417, upper bound: 0.0012664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013360, upper bound: 0.0012734
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0013453
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012588, upper bound: 0.0013492
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013204, upper bound: 0.0012937
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0013034
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012681, upper bound: 0.0013530
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0013579
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012574, upper bound: 0.0012963
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013059, upper bound: 0.0013057
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013028, upper bound: 0.0013059
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012935, upper bound: 0.0013134
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013574, upper bound: 0.0012576
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013528, upper bound: 0.0012685
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012993, upper bound: 0.0013123
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012919, upper bound: 0.0013204
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013485, upper bound: 0.0012590
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013450, upper bound: 0.0012698
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012731, upper bound: 0.0013365
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012663, upper bound: 0.0013427
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013293, upper bound: 0.0012894
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013250, upper bound: 0.0013021
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012712, upper bound: 0.0013452
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0012649, upper bound: 0.0013517
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013222, upper bound: 0.0012923
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.90
Output dim: 0, lower bound: -0.0013183, upper bound: 0.0013045

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018052, 0.0018973
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004498, 0.0004728
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025054, 0.0023838
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010850, 0.0011403
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004849, 0.0004614
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031511, 0.0029982
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007610, 0.0007998
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019688, 0.0020693
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010354, 0.0010882
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012618, 0.0012006

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012768, upper bound: 0.0012800
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012652, upper bound: 0.0012898
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017927, 0.0018955
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004467, 0.0004723
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025029, 0.0023673
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010775, 0.0011392
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004844, 0.0004582
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031480, 0.0029774
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007557, 0.0007990
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019552, 0.0020673
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010282, 0.0010872
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012606, 0.0011923

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012639, upper bound: 0.0012814
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012554, upper bound: 0.0012944
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018856, 0.0017992
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004699, 0.0004483
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023758, 0.0024900
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011333, 0.0010814
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004598, 0.0004819
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029881, 0.0031317
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007949, 0.0007584
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020566, 0.0019623
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010815, 0.0010319
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011966, 0.0012541

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013226, upper bound: 0.0012243
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013147, upper bound: 0.0012372
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018792, 0.0017995
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004682, 0.0004484
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023762, 0.0024815
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011295, 0.0010815
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004599, 0.0004803
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029886, 0.0031210
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007921, 0.0007585
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020495, 0.0019626
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010778, 0.0010321
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011968, 0.0012498

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013155, upper bound: 0.0012276
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013095, upper bound: 0.0012446
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018032, 0.0019024
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004493, 0.0004740
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025121, 0.0023811
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010838, 0.0011434
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004862, 0.0004609
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031595, 0.0029948
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007601, 0.0008019
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019667, 0.0020748
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010343, 0.0010911
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012652, 0.0011993

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0012867
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012621, upper bound: 0.0012968
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017902, 0.0019009
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004461, 0.0004736
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025101, 0.0023640
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010760, 0.0011425
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004858, 0.0004575
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031570, 0.0029733
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007546, 0.0008013
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019525, 0.0020732
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010268, 0.0010903
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012642, 0.0011906

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012610, upper bound: 0.0012892
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012524, upper bound: 0.0013014
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018809, 0.0018017
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004687, 0.0004489
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023792, 0.0024837
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011305, 0.0010829
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004605, 0.0004807
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029924, 0.0031238
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007929, 0.0007595
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020514, 0.0019650
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010788, 0.0010334
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011983, 0.0012509

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013135, upper bound: 0.0012253
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013041, upper bound: 0.0012387
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018741, 0.0018024
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004670, 0.0004491
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023800, 0.0024747
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011264, 0.0010833
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004606, 0.0004790
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029934, 0.0031125
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007900, 0.0007598
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020440, 0.0019657
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010749, 0.0010338
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011987, 0.0012464

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013066, upper bound: 0.0012294
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012996, upper bound: 0.0012464
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017511, 0.0019250
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004363, 0.0004797
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025419, 0.0023123
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010524, 0.0011570
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004920, 0.0004475
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031971, 0.0029082
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007381, 0.0008115
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019098, 0.0020995
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010043, 0.0011041
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012803, 0.0011646

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0013067
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012257, upper bound: 0.0013149
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017412, 0.0019281
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004339, 0.0004804
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025461, 0.0022992
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010465, 0.0011589
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004928, 0.0004450
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032023, 0.0028918
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007340, 0.0008128
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018990, 0.0021029
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009987, 0.0011059
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012823, 0.0011580

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012310, upper bound: 0.0013094
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012189, upper bound: 0.0013188
time: 1.36 seconds

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

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012538
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012632
time: 1.09 seconds

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

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012600
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0012708
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017488, 0.0019305
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004358, 0.0004810
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025491, 0.0023093
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010511, 0.0011603
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004934, 0.0004470
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032062, 0.0029045
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007372, 0.0008138
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019073, 0.0021054
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010030, 0.0011072
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012839, 0.0011631

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012413, upper bound: 0.0013158
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013232
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017393, 0.0019337
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004334, 0.0004818
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025535, 0.0022967
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010454, 0.0011622
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004942, 0.0004445
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032116, 0.0028887
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007332, 0.0008151
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018970, 0.0021090
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009976, 0.0011091
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012861, 0.0011568

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012292, upper bound: 0.0013190
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0013282
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018307, 0.0018327
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004562, 0.0004567
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024201, 0.0024174
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011003, 0.0011015
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004684, 0.0004679
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030438, 0.0030405
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007717, 0.0007726
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019966, 0.0019988
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010500, 0.0010512
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012189, 0.0012175

Time for backsubstitution: 1.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012853, upper bound: 0.0012565
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0012650
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018247, 0.0018363
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004547, 0.0004575
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024248, 0.0024095
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010967, 0.0011036
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004693, 0.0004664
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030497, 0.0030305
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007692, 0.0007740
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019901, 0.0020027
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010466, 0.0010532
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012212, 0.0012136

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012769, upper bound: 0.0012641
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012679, upper bound: 0.0012743
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019129, 0.0018213
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004767, 0.0004538
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024049, 0.0025260
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011497, 0.0010946
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004655, 0.0004889
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030248, 0.0031771
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008064, 0.0007677
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020863, 0.0019863
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010972, 0.0010446
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012113, 0.0012722

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012778, upper bound: 0.0012679
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012662, upper bound: 0.0012769
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019004, 0.0018273
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004735, 0.0004553
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024129, 0.0025095
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011422, 0.0010982
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004670, 0.0004857
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030348, 0.0031563
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008011, 0.0007703
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020727, 0.0019929
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010900, 0.0010480
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012153, 0.0012639

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012676, upper bound: 0.0012729
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012583, upper bound: 0.0012853
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019934, 0.0017354
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004967, 0.0004324
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022915, 0.0026322
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011981, 0.0010430
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004435, 0.0005095
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028821, 0.0033106
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008403, 0.0007315
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021741, 0.0018927
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011433, 0.0009953
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011541, 0.0013257

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013286, upper bound: 0.0012176
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013193, upper bound: 0.0012290
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019869, 0.0017443
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004951, 0.0004346
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023033, 0.0026237
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011942, 0.0010484
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004458, 0.0005078
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028970, 0.0032999
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008376, 0.0007353
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021670, 0.0019024
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011396, 0.0010005
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011601, 0.0013214

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013233, upper bound: 0.0012243
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013160, upper bound: 0.0012408
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019109, 0.0018259
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004762, 0.0004550
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024111, 0.0025234
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011485, 0.0010974
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004667, 0.0004884
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030325, 0.0031738
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008055, 0.0007697
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020842, 0.0019914
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010960, 0.0010473
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012143, 0.0012709

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012756, upper bound: 0.0012743
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012629, upper bound: 0.0012833
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018980, 0.0018322
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004729, 0.0004565
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024194, 0.0025062
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011407, 0.0011012
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004683, 0.0004851
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030429, 0.0031522
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008001, 0.0007723
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020700, 0.0019982
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010886, 0.0010509
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012185, 0.0012623

Time for backsubstitution: 1.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012651, upper bound: 0.0012799
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012551, upper bound: 0.0012921
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019886, 0.0017372
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004955, 0.0004329
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022940, 0.0026259
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011952, 0.0010441
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004440, 0.0005082
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028852, 0.0033027
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008383, 0.0007323
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021689, 0.0018947
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011406, 0.0009964
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011554, 0.0013226

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013197, upper bound: 0.0012189
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013100, upper bound: 0.0012309
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019818, 0.0017471
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004938, 0.0004353
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023071, 0.0026170
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011911, 0.0010501
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004465, 0.0005065
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029017, 0.0032914
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008354, 0.0007365
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021614, 0.0019055
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011367, 0.0010021
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011620, 0.0013180

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013152, upper bound: 0.0012257
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013071, upper bound: 0.0012426
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018588, 0.0018691
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004632, 0.0004657
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024681, 0.0024545
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011172, 0.0011234
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004777, 0.0004751
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031043, 0.0030871
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007836, 0.0007879
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020273, 0.0020385
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010661, 0.0010720
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012431, 0.0012362

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012466, upper bound: 0.0012987
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012295, upper bound: 0.0013061
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018489, 0.0018760
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004607, 0.0004674
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024772, 0.0024415
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011113, 0.0011275
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004795, 0.0004725
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031157, 0.0030708
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007794, 0.0007908
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020165, 0.0020460
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010605, 0.0010760
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012477, 0.0012297

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012387, upper bound: 0.0013034
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012252, upper bound: 0.0013122
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019435, 0.0017842
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004843, 0.0004446
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023561, 0.0025664
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011681, 0.0010724
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004560, 0.0004967
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029633, 0.0032278
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008193, 0.0007521
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021197, 0.0019460
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011147, 0.0010234
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011866, 0.0012926

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013014, upper bound: 0.0012507
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012892, upper bound: 0.0012580
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019371, 0.0017972
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004827, 0.0004478
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023732, 0.0025579
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011643, 0.0010802
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004593, 0.0004951
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029848, 0.0032172
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008166, 0.0007576
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021127, 0.0019601
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011110, 0.0010308
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011953, 0.0012883

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012968, upper bound: 0.0012590
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012867, upper bound: 0.0012689
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018565, 0.0018744
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004626, 0.0004670
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024751, 0.0024515
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011158, 0.0011266
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004791, 0.0004745
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031130, 0.0030834
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007826, 0.0007901
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020248, 0.0020443
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010648, 0.0010751
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012466, 0.0012347

Time for backsubstitution: 1.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012450, upper bound: 0.0013093
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012277, upper bound: 0.0013151
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018470, 0.0018808
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004602, 0.0004686
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024836, 0.0024390
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011101, 0.0011304
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004807, 0.0004721
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031237, 0.0030676
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007786, 0.0007928
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020144, 0.0020513
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010594, 0.0010787
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012509, 0.0012284

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012374, upper bound: 0.0013139
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013217
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019384, 0.0017867
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004830, 0.0004452
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023593, 0.0025597
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011650, 0.0010739
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004566, 0.0004954
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029674, 0.0032194
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008171, 0.0007532
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021141, 0.0019487
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011118, 0.0010248
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011883, 0.0012892

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0012530
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012814, upper bound: 0.0012607
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019324, 0.0017996
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004815, 0.0004484
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023764, 0.0025518
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011615, 0.0010816
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004599, 0.0004939
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029889, 0.0032095
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008146, 0.0007586
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021076, 0.0019627
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011084, 0.0010322
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011969, 0.0012852

Time for backsubstitution: 1.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012898, upper bound: 0.0012627
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012800, upper bound: 0.0012729
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017996, 0.0019069
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004484, 0.0004751
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025180, 0.0023764
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010816, 0.0011461
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004874, 0.0004599
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031670, 0.0029889
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007586, 0.0008038
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019627, 0.0020797
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010322, 0.0010937
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012682, 0.0011969

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012729, upper bound: 0.0012800
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012627, upper bound: 0.0012898
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017867, 0.0019050
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004452, 0.0004747
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025156, 0.0023593
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010739, 0.0011450
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004869, 0.0004566
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031639, 0.0029674
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007532, 0.0008030
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019487, 0.0020777
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010248, 0.0010926
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012670, 0.0011883

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012607, upper bound: 0.0012814
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012944
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013217, upper bound: 0.0012243
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0012374
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013151, upper bound: 0.0012277
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013093, upper bound: 0.0012450
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017972, 0.0019120
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004478, 0.0004764
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025247, 0.0023732
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010802, 0.0011491
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004887, 0.0004593
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031754, 0.0029848
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007576, 0.0008060
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019601, 0.0020853
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010308, 0.0010966
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012716, 0.0011953

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012867
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012590, upper bound: 0.0012968
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017842, 0.0019104
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004446, 0.0004760
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025227, 0.0023561
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010724, 0.0011482
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004883, 0.0004560
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031729, 0.0029633
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007521, 0.0008053
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019460, 0.0020836
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010234, 0.0010958
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012706, 0.0011866

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012580, upper bound: 0.0012892
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012507, upper bound: 0.0013014
time: 1.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018760, 0.0018113
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004674, 0.0004513
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023918, 0.0024772
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011275, 0.0010887
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004629, 0.0004795
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030083, 0.0031157
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007908, 0.0007635
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020460, 0.0019755
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010760, 0.0010389
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012046, 0.0012477

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0012252
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013033, upper bound: 0.0012387
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018691, 0.0018119
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004657, 0.0004515
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023927, 0.0024681
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011234, 0.0010890
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004631, 0.0004777
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030093, 0.0031043
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007879, 0.0007638
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020385, 0.0019762
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010720, 0.0010393
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012051, 0.0012431

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0012295
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012987, upper bound: 0.0012466
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017471, 0.0019346
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004353, 0.0004820
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025546, 0.0023071
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010501, 0.0011627
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004944, 0.0004465
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032130, 0.0029017
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007365, 0.0008155
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019055, 0.0021099
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010021, 0.0011096
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012866, 0.0011620

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012426, upper bound: 0.0013071
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012257, upper bound: 0.0013152
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017372, 0.0019377
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004329, 0.0004828
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025587, 0.0022940
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010441, 0.0011646
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004952, 0.0004440
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032182, 0.0028852
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007323, 0.0008168
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018947, 0.0021133
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009964, 0.0011114
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012887, 0.0011554

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012309, upper bound: 0.0013100
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0013197
time: 1.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012551
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012651
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012629
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012755
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017443, 0.0019400
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004346, 0.0004834
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025618, 0.0023033
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010484, 0.0011660
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004958, 0.0004458
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032221, 0.0028970
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007353, 0.0008178
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019024, 0.0021159
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010005, 0.0011127
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012903, 0.0011601

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012408, upper bound: 0.0013160
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013233
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017354, 0.0019433
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004324, 0.0004842
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0025661, 0.0022915
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010430, 0.0011680
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004967, 0.0004435
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0032275, 0.0028821
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007315, 0.0008192
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018927, 0.0021195
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009953, 0.0011146
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012924, 0.0011541

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012290, upper bound: 0.0013193
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0013286
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018273, 0.0018423
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004553, 0.0004590
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024327, 0.0024129
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010982, 0.0011073
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004708, 0.0004670
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030597, 0.0030348
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007703, 0.0007766
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019929, 0.0020093
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010480, 0.0010567
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012252, 0.0012153

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012853, upper bound: 0.0012583
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012676
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018213, 0.0018458
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004538, 0.0004599
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024374, 0.0024049
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010946, 0.0011094
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004718, 0.0004655
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030656, 0.0030248
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007677, 0.0007781
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019863, 0.0020131
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010446, 0.0010587
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012276, 0.0012113

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012769, upper bound: 0.0012662
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012679, upper bound: 0.0012778
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019073, 0.0018308
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004753, 0.0004562
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024176, 0.0025186
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011464, 0.0011004
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004679, 0.0004875
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030407, 0.0031678
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008040, 0.0007718
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020802, 0.0019968
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010940, 0.0010501
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012176, 0.0012685

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012743, upper bound: 0.0012679
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012641, upper bound: 0.0012769
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018944, 0.0018369
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004720, 0.0004577
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024255, 0.0025016
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011386, 0.0011040
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004695, 0.0004842
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030507, 0.0031463
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007986, 0.0007743
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020661, 0.0020033
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010866, 0.0010535
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012216, 0.0012599

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012650, upper bound: 0.0012729
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012565, upper bound: 0.0012853
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019885, 0.0017449
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004955, 0.0004348
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023042, 0.0026258
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011952, 0.0010488
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004460, 0.0005082
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028980, 0.0033026
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008382, 0.0007356
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021687, 0.0019031
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011405, 0.0010008
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011605, 0.0013225

Time for backsubstitution: 1.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013282, upper bound: 0.0012177
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013190, upper bound: 0.0012292
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019821, 0.0017539
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004939, 0.0004370
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023160, 0.0026174
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011913, 0.0010541
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004483, 0.0005066
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029129, 0.0032919
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008355, 0.0007393
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021618, 0.0019129
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011369, 0.0010060
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011665, 0.0013182

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012243
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013158, upper bound: 0.0012413
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019049, 0.0018355
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004747, 0.0004574
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024237, 0.0025154
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011449, 0.0011032
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004691, 0.0004869
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030484, 0.0031637
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008030, 0.0007737
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020776, 0.0020018
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010926, 0.0010527
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012207, 0.0012669

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012708, upper bound: 0.0012743
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012600, upper bound: 0.0012833
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018920, 0.0018417
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004714, 0.0004589
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024320, 0.0024983
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011371, 0.0011069
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004707, 0.0004835
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030588, 0.0031422
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007975, 0.0007764
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020634, 0.0020087
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010851, 0.0010563
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012249, 0.0012583

Time for backsubstitution: 1.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012632, upper bound: 0.0012799
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012538, upper bound: 0.0012921
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019837, 0.0017468
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004943, 0.0004353
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023066, 0.0026195
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011923, 0.0010499
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004464, 0.0005070
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029012, 0.0032946
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008362, 0.0007363
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021635, 0.0019051
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011378, 0.0010019
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011617, 0.0013193

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013188, upper bound: 0.0012189
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0012310
time: 1.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019768, 0.0017567
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004926, 0.0004377
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023197, 0.0026104
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011881, 0.0010558
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004490, 0.0005052
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029176, 0.0032832
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008333, 0.0007405
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021560, 0.0019159
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011338, 0.0010076
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011683, 0.0013147

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013149, upper bound: 0.0012257
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013067, upper bound: 0.0012428
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018549, 0.0018787
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004622, 0.0004681
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024808, 0.0024493
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011148, 0.0011291
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004801, 0.0004741
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031202, 0.0030806
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007819, 0.0007919
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020230, 0.0020490
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010639, 0.0010775
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012495, 0.0012336

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012464, upper bound: 0.0012996
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012294, upper bound: 0.0013066
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018450, 0.0018856
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004597, 0.0004698
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024899, 0.0024362
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011089, 0.0011333
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004819, 0.0004715
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031316, 0.0030642
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007777, 0.0007948
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020122, 0.0020565
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010582, 0.0010815
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012540, 0.0012270

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012387, upper bound: 0.0013041
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012252, upper bound: 0.0013135
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019399, 0.0017938
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004834, 0.0004470
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023687, 0.0025616
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011659, 0.0010781
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004585, 0.0004958
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029792, 0.0032218
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008177, 0.0007562
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021157, 0.0019564
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011126, 0.0010289
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011930, 0.0012902

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013014, upper bound: 0.0012524
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012892, upper bound: 0.0012610
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019336, 0.0018068
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004818, 0.0004502
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023858, 0.0025533
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011622, 0.0010859
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004618, 0.0004942
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030007, 0.0032114
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008151, 0.0007616
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021089, 0.0019705
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011090, 0.0010363
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012016, 0.0012860

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012968, upper bound: 0.0012621
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012867, upper bound: 0.0012741
time: 1.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018520, 0.0018840
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004615, 0.0004694
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024878, 0.0024456
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011131, 0.0011323
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004815, 0.0004733
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031289, 0.0030759
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007807, 0.0007942
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020199, 0.0020547
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010622, 0.0010806
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012530, 0.0012317

Time for backsubstitution: 1.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012446, upper bound: 0.0013095
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012276, upper bound: 0.0013155
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0018431, 0.0018904
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004592, 0.0004710
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0024962, 0.0024338
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011077, 0.0011362
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004831, 0.0004711
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0031396, 0.0030610
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007769, 0.0007969
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0020101, 0.0020617
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010571, 0.0010842
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012572, 0.0012258

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012372, upper bound: 0.0013147
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013226
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019350, 0.0017963
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004822, 0.0004476
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023720, 0.0025551
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011630, 0.0010796
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004591, 0.0004945
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029833, 0.0032137
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008157, 0.0007572
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021104, 0.0019591
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011098, 0.0010303
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011947, 0.0012869

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0012554
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012814, upper bound: 0.0012639
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0019290, 0.0018092
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004806, 0.0004508
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023890, 0.0025472
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0011594, 0.0010874
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004624, 0.0004930
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030048, 0.0032037
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0008131, 0.0007626
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0021038, 0.0019732
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0011064, 0.0010377
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012032, 0.0012829

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012898, upper bound: 0.0012652
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012800, upper bound: 0.0012768
time: 1.32 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012768, upper bound: 0.0012800
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012652, upper bound: 0.0012898
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012639, upper bound: 0.0012814
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012554, upper bound: 0.0012944
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013226, upper bound: 0.0012243
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013147, upper bound: 0.0012372
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013155, upper bound: 0.0012276
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013095, upper bound: 0.0012446
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0012867
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012621, upper bound: 0.0012968
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012610, upper bound: 0.0012892
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012524, upper bound: 0.0013014
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013135, upper bound: 0.0012253
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013041, upper bound: 0.0012387
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013066, upper bound: 0.0012294
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012996, upper bound: 0.0012464
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012428, upper bound: 0.0013067
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012257, upper bound: 0.0013149
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012310, upper bound: 0.0013094
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012189, upper bound: 0.0013188
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012538
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012632
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012600
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0012708
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012413, upper bound: 0.0013158
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013232
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012292, upper bound: 0.0013190
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0013282
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012853, upper bound: 0.0012565
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012177, upper bound: 0.0012650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012769, upper bound: 0.0012641
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012679, upper bound: 0.0012743
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012778, upper bound: 0.0012679
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012662, upper bound: 0.0012769
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012676, upper bound: 0.0012729
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012583, upper bound: 0.0012853
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013286, upper bound: 0.0012176
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013193, upper bound: 0.0012290
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013233, upper bound: 0.0012243
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013160, upper bound: 0.0012408
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012756, upper bound: 0.0012743
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012629, upper bound: 0.0012833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012651, upper bound: 0.0012799
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012551, upper bound: 0.0012921
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013197, upper bound: 0.0012189
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013100, upper bound: 0.0012309
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013152, upper bound: 0.0012257
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013071, upper bound: 0.0012426
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012466, upper bound: 0.0012987
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012295, upper bound: 0.0013061
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012387, upper bound: 0.0013034
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012252, upper bound: 0.0013122
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013014, upper bound: 0.0012507
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012892, upper bound: 0.0012580
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012968, upper bound: 0.0012590
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012867, upper bound: 0.0012689
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012450, upper bound: 0.0013093
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012277, upper bound: 0.0013151
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012374, upper bound: 0.0013139
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013217
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0012530
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012814, upper bound: 0.0012607
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012898, upper bound: 0.0012627
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012800, upper bound: 0.0012729
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012729, upper bound: 0.0012800
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012627, upper bound: 0.0012898
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012607, upper bound: 0.0012814
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012944
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013217, upper bound: 0.0012243
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013139, upper bound: 0.0012374
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013151, upper bound: 0.0012277
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013093, upper bound: 0.0012450
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012867
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012590, upper bound: 0.0012968
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012580, upper bound: 0.0012892
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012507, upper bound: 0.0013014
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013122, upper bound: 0.0012252
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013033, upper bound: 0.0012387
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013060, upper bound: 0.0012295
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012987, upper bound: 0.0012466
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012426, upper bound: 0.0013071
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012257, upper bound: 0.0013152
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012309, upper bound: 0.0013100
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0013197
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012921, upper bound: 0.0012551
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012799, upper bound: 0.0012651
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012833, upper bound: 0.0012629
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012755
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012408, upper bound: 0.0013160
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013233
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012290, upper bound: 0.0013193
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0013286
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012853, upper bound: 0.0012583
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012176, upper bound: 0.0012676
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012769, upper bound: 0.0012662
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012679, upper bound: 0.0012778
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012743, upper bound: 0.0012679
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012641, upper bound: 0.0012769
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012650, upper bound: 0.0012729
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012565, upper bound: 0.0012853
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013282, upper bound: 0.0012177
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013190, upper bound: 0.0012292
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013232, upper bound: 0.0012243
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013158, upper bound: 0.0012413
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012708, upper bound: 0.0012743
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012600, upper bound: 0.0012833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012632, upper bound: 0.0012799
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012538, upper bound: 0.0012921
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013188, upper bound: 0.0012189
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013094, upper bound: 0.0012310
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013149, upper bound: 0.0012257
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013067, upper bound: 0.0012428
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012464, upper bound: 0.0012996
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012294, upper bound: 0.0013066
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012387, upper bound: 0.0013041
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012252, upper bound: 0.0013135
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0013014, upper bound: 0.0012524
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012892, upper bound: 0.0012610
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012968, upper bound: 0.0012621
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012867, upper bound: 0.0012741
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012446, upper bound: 0.0013095
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012276, upper bound: 0.0013155
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012372, upper bound: 0.0013147
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012243, upper bound: 0.0013226
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012944, upper bound: 0.0012554
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012814, upper bound: 0.0012639
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012898, upper bound: 0.0012652
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.10
Output dim: 0, lower bound: -0.0012800, upper bound: 0.0012768

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017224, 0.0018002
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004292, 0.0004486
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023772, 0.0022744
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010352, 0.0010820
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004601, 0.0004402
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029899, 0.0028606
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007260, 0.0007589
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018785, 0.0019634
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009879, 0.0010325
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011973, 0.0011455

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011470, upper bound: 0.0011486
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011476, upper bound: 0.0011486
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017081, 0.0018110
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004256, 0.0004513
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023914, 0.0022556
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010266, 0.0010885
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004629, 0.0004366
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030078, 0.0028369
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007200, 0.0007634
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018630, 0.0019751
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009797, 0.0010387
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012044, 0.0011360

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011357, upper bound: 0.0011579
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011385, upper bound: 0.0011579
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017084, 0.0017984
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004257, 0.0004481
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023747, 0.0022559
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010268, 0.0010809
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004596, 0.0004366
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029868, 0.0028374
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007202, 0.0007581
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018632, 0.0019614
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009799, 0.0010315
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011960, 0.0011362

Time for backsubstitution: 1.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010861, upper bound: 0.0011495
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011358, upper bound: 0.0011495
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0016956, 0.0018117
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004225, 0.0004514
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023923, 0.0022391
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010191, 0.0010889
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004630, 0.0004334
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030089, 0.0028162
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007148, 0.0007637
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018493, 0.0019759
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009725, 0.0010391
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012049, 0.0011277

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010861, upper bound: 0.0011615
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010861, upper bound: 0.0011615
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011923, upper bound: 0.0010935
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011946, upper bound: 0.0010934
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011833, upper bound: 0.0011054
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011887, upper bound: 0.0011054
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 1.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011849, upper bound: 0.0010967
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010861, upper bound: 0.0010967
time: 1.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010861, upper bound: 0.0011117
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011841, upper bound: 0.0011117
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017204, 0.0018053
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004287, 0.0004498
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023839, 0.0022718
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010340, 0.0010850
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004614, 0.0004397
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029983, 0.0028573
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007252, 0.0007610
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018763, 0.0019689
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009867, 0.0010354
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012007, 0.0011442

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011425, upper bound: 0.0011580
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011425, upper bound: 0.0011557
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017062, 0.0018158
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004251, 0.0004525
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023978, 0.0022530
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010254, 0.0010914
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004641, 0.0004361
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030158, 0.0028336
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007192, 0.0007654
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018608, 0.0019804
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009786, 0.0010415
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012076, 0.0011347

Time for backsubstitution: 1.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011324, upper bound: 0.0011674
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011333, upper bound: 0.0011661
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017061, 0.0018038
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004251, 0.0004495
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023819, 0.0022529
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010254, 0.0010841
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004610, 0.0004360
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0029958, 0.0028335
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007192, 0.0007604
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018607, 0.0019673
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009785, 0.0010346
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011996, 0.0011347

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011305, upper bound: 0.0011596
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011309, upper bound: 0.0011585
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0016932, 0.0018167
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004219, 0.0004527
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0023989, 0.0022358
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010176, 0.0010919
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004643, 0.0004327
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0030172, 0.0028120
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007137, 0.0007658
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0018466, 0.0019814
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0009711, 0.0010420
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0012082, 0.0011261

Time for backsubstitution: 1.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011232, upper bound: 0.0011720
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011249, upper bound: 0.0011707
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017946, 0.0017047
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004472, 0.0004248
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022510, 0.0023698
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010786, 0.0010246
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004357, 0.0004587
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028311, 0.0029805
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007565, 0.0007186
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019573, 0.0018592
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010293, 0.0009777
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011337, 0.0011935

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011824, upper bound: 0.0010965
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011825, upper bound: 0.0010952
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017838, 0.0017172
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004445, 0.0004279
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022676, 0.0023555
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010721, 0.0010321
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004389, 0.0004559
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028520, 0.0029626
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007519, 0.0007239
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019455, 0.0018729
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010231, 0.0009849
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011421, 0.0011864

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011746, upper bound: 0.0011087
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011758, upper bound: 0.0011082
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017862, 0.0017053
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004451, 0.0004249
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022518, 0.0023587
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010736, 0.0010249
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004358, 0.0004565
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028322, 0.0029666
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007530, 0.0007188
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019481, 0.0018599
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010245, 0.0009781
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011341, 0.0011880

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011761, upper bound: 0.0011007
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010861, upper bound: 0.0010995
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9887190, 0.9922218, 0.9887190, 0.9922218, -0.0017770, 0.0017188
1: -0.0040749, -0.0032021, -0.0040749, -0.0032021, -0.0004428, 0.0004283
2: 0.0069154, 0.0115406, 0.0069154, 0.0115406, -0.0022697, 0.0023465
3: -0.0065259, -0.0044207, -0.0065259, -0.0044207, -0.0010680, 0.0010331
4: 0.0018663, 0.0027616, 0.0018663, 0.0027616, -0.0004393, 0.0004542
5: 0.0076571, 0.0134745, 0.0076571, 0.0134745, -0.0028547, 0.0029513
6: -0.0018791, -0.0004026, -0.0018791, -0.0004026, -0.0007491, 0.0007245
7: -0.0079996, -0.0041794, -0.0079996, -0.0041794, -0.0019381, 0.0018746
8: -0.0037710, -0.0017620, -0.0037710, -0.0017620, -0.0010192, 0.0009858
9: 0.0001793, 0.0025088, 0.0001793, 0.0025088, -0.0011431, 0.0011818

Time for backsubstitution: 1.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Candidate
type: DSZ, layer: 1, pos: 102

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 41
type: DSZ, layer: 3, pos: 54
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 67
type: DSZ, layer: 3, pos: 136
type: DSZ, layer: 3, pos: 139
type: DSZ, layer: 3, pos: 144
type: DSZ, layer: 3, pos: 145
type: DSZ, layer: 3, pos: 152
type: DSZ, layer: 3, pos: 165
type: DSZ, layer: 3, pos: 174
type: DSZ, layer: 3, pos: 241
type: DSZ, layer: 3, pos: 248

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 3, pos: 41

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011703, upper bound: 0.0011161
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0011716, upper bound: 0.0011158
time: 1.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 1.52 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.40 + 597.06 = 600.47 seconds
