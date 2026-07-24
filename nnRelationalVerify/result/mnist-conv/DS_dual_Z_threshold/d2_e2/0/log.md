## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.32452505010000005


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6905186, 0.6905186)
1: (-4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5279188, 0.5279183)
2: (10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5151484, 0.5151484)
3: (-3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6740751, 0.6740749)
4: (-6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5155964, 0.5155964)
5: (-10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6018240, 0.6018243)
6: (-13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6701522, 0.6701524)
7: (-4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6258245, 0.6258245)
8: (-4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4052844, 0.4052843)
9: (-10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6959014, 0.6959014)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 21.68 + 36.96 = 58.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3248499, upper bound: 0.3248496

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 4637
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4653

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248440, upper bound: 0.3226901
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3226902, upper bound: 0.3248444
time: 5.45 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.95
Output dim: 2, lower bound: -0.3248440, upper bound: 0.3226901
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.95
Output dim: 2, lower bound: -0.3226902, upper bound: 0.3248444

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6904941, 0.6904550
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5268273, 0.5249937
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5139228, 0.5118647
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6725750, 0.6735163
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5068552, 0.5123448
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6014304, 0.6007657
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6683469, 0.6653218
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6243811, 0.6252856
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4051309, 0.4048741
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6858549, 0.6921530

Time for backsubstitution: 20.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 4637
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4615

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248393, upper bound: 0.3204093
time: 4.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3225623, upper bound: 0.3226862
time: 3.99 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6904550, 0.6904941
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5249934, 0.5268273
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5118648, 0.5139229
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6735163, 0.6725750
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5123451, 0.5068550
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6007657, 0.6014302
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6653218, 0.6683469
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6252856, 0.6243811
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4048741, 0.4051310
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6921530, 0.6858547

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 4637
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4615

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226855, upper bound: 0.3225623
time: 7.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3204086, upper bound: 0.3248392
time: 6.89 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 35.89 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 35.89
Output dim: 2, lower bound: -0.3248393, upper bound: 0.3204093
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 35.89
Output dim: 2, lower bound: -0.3225623, upper bound: 0.3226862
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 35.89
Output dim: 2, lower bound: -0.3226855, upper bound: 0.3225623
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 35.89
Output dim: 2, lower bound: -0.3204086, upper bound: 0.3248392

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6924720, 0.6934478
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5206993, 0.5219924
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5074102, 0.5030477
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6671615, 0.6701381
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5008016, 0.5042701
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5966358, 0.5974884
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6692343, 0.6654422
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6179171, 0.6166732
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4027085, 0.4030561
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6822715, 0.6873724

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4637
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4637

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248300, upper bound: 0.3178698
time: 5.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3223001, upper bound: 0.3203993
time: 5.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6934476, 0.6924717
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5219924, 0.5206990
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5030476, 0.5074103
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6701384, 0.6671615
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5042701, 0.5008018
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5974884, 0.5966358
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6654425, 0.6692340
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6166735, 0.6179171
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4030561, 0.4027085
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6873727, 0.6822715

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4637
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4637

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3203993, upper bound: 0.3222999
time: 6.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3178692, upper bound: 0.3248304
time: 5.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.04 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.04
Output dim: 2, lower bound: -0.3248300, upper bound: 0.3178698
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 34.04
Output dim: 2, lower bound: -0.3223001, upper bound: 0.3203993
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 34.04
Output dim: 2, lower bound: -0.3203993, upper bound: 0.3222999
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.04
Output dim: 2, lower bound: -0.3178692, upper bound: 0.3248304

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6909289, 0.6929114
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5192468, 0.5214901
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5064504, 0.5002837
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6671157, 0.6700051
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5000107, 0.5039952
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5950310, 0.5969341
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6658397, 0.6642613
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6169295, 0.6138232
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4001999, 0.4021842
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6804872, 0.6867523

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6224

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246233, upper bound: 0.3178688
time: 7.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248295, upper bound: 0.3176629
time: 4.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6929116, 0.6909292
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5214899, 0.5192468
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5002835, 0.5064504
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6700053, 0.6671157
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5039952, 0.5000110
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5969341, 0.5950308
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6642613, 0.6658392
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6138234, 0.6169295
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4021842, 0.4001999
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6867523, 0.6804872

Time for backsubstitution: 21.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6224

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3176624, upper bound: 0.3248302
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3178688, upper bound: 0.3246231
time: 8.17 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 35.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.06
Output dim: 2, lower bound: -0.3246233, upper bound: 0.3178688
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.06
Output dim: 2, lower bound: -0.3248295, upper bound: 0.3176629
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 35.06
Output dim: 2, lower bound: -0.3176624, upper bound: 0.3248302
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 35.06
Output dim: 2, lower bound: -0.3178688, upper bound: 0.3246231

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6863503, 0.6907411
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5154169, 0.5196750
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5058714, 0.4990623
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6664586, 0.6696935
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4959011, 0.5020477
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5949454, 0.5967543
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6619320, 0.6560295
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6109638, 0.6109960
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3977890, 0.3971031
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6717563, 0.6826067

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 4546

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3237668, upper bound: 0.3178670
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246208, upper bound: 0.3170125
time: 5.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6887589, 0.6883323
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5174320, 0.5176599
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5052291, 0.4997046
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6668038, 0.6693482
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4980636, 0.4998852
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5948510, 0.5968485
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6576076, 0.6603541
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6141024, 0.6078575
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3951187, 0.3997735
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6763415, 0.6780210

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 4546

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3239730, upper bound: 0.3176606
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3168061
time: 5.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6883321, 0.6887589
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5176599, 0.5174320
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.4997045, 0.5052290
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6693482, 0.6668038
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4998851, 0.4980634
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5968485, 0.5948510
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6603541, 0.6576073
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6078577, 0.6141024
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3997734, 0.3951187
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6780210, 0.6763415

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 4546

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3168061, upper bound: 0.3248278
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3176600, upper bound: 0.3239737
time: 4.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6907411, 0.6863501
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5196750, 0.5154166
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.4990622, 0.5058711
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6696935, 0.6664586
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5020480, 0.4959010
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5967546, 0.5949454
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6560292, 0.6619320
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6109962, 0.6109638
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3971031, 0.3977890
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6826067, 0.6717558

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 4546

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3170124, upper bound: 0.3246214
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3178663, upper bound: 0.3237668
time: 5.47 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 33.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3237668, upper bound: 0.3178670
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3246208, upper bound: 0.3170125
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3239730, upper bound: 0.3176606
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3168061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3168061, upper bound: 0.3248278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3176600, upper bound: 0.3239737
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3170124, upper bound: 0.3246214
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 33.23
Output dim: 2, lower bound: -0.3178663, upper bound: 0.3237668

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6863503, 0.6897404
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5138245, 0.5196750
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5058714, 0.4981513
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6664586, 0.6691616
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4957078, 0.5020477
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5937243, 0.5967543
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6589122, 0.6560295
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6109638, 0.6104891
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3977890, 0.3964285
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6717563, 0.6789980

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3241546, upper bound: 0.3170131
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246208, upper bound: 0.3165467
time: 5.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6887589, 0.6873317
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5158396, 0.5176599
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5052291, 0.4987934
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6668038, 0.6688163
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4978702, 0.4998852
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5936298, 0.5968485
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6545873, 0.6603541
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6141024, 0.6073506
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3951187, 0.3990989
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6763415, 0.6744123

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3243609, upper bound: 0.3168060
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3163402
time: 3.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6873317, 0.6893580
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5186157, 0.5158393
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.4987932, 0.5057738
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6688161, 0.6671214
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4999993, 0.4978701
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5975800, 0.5936303
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6621642, 0.6545873
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6073503, 0.6144030
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3990989, 0.3955210
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6744123, 0.6785035

Time for backsubstitution: 21.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3163396, upper bound: 0.3248278
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3168061, upper bound: 0.3243615
time: 4.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6897402, 0.6869493
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5206308, 0.5138242
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.4981509, 0.5064158
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6691613, 0.6667762
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5021617, 0.4957076
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5974860, 0.5937243
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6578393, 0.6589122
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6104889, 0.6112645
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3964283, 0.3981913
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6789980, 0.6739180

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3165460, upper bound: 0.3246214
time: 4.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3170124, upper bound: 0.3241551
time: 4.61 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 30.74 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3241546, upper bound: 0.3170131
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3246208, upper bound: 0.3165467
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3243609, upper bound: 0.3168060
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3163402
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3163396, upper bound: 0.3248278
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3168061, upper bound: 0.3243615
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3165460, upper bound: 0.3246214
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 30.74
Output dim: 2, lower bound: -0.3170124, upper bound: 0.3241551

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6863565, 0.6897452
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5138230, 0.5196745
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5058718, 0.4981508
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6664586, 0.6691616
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4957080, 0.5020480
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5937262, 0.5967565
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6589084, 0.6560259
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6109614, 0.6104865
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3977847, 0.3964226
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6717529, 0.6789937

Time for backsubstitution: 21.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3233785, upper bound: 0.3139618
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3220374, upper bound: 0.3153023
time: 4.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6887655, 0.6873364
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5158381, 0.5176592
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5052295, 0.4987931
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6668038, 0.6688163
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.4978704, 0.4998856
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5936317, 0.5968506
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6545835, 0.6603508
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6141000, 0.6073477
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3951144, 0.3990930
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6763387, 0.6744084

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3235848, upper bound: 0.3137554
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222437, upper bound: 0.3150958
time: 3.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6873360, 0.6893640
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5186148, 0.5158384
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.4987931, 0.5057735
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6688166, 0.6671216
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5000000, 0.4978704
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5975823, 0.5936317
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6621614, 0.6545835
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6073480, 0.6144011
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3990932, 0.3955165
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6744084, 0.6785004

Time for backsubstitution: 21.84 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.63 + 551.77 = 610.40 seconds
