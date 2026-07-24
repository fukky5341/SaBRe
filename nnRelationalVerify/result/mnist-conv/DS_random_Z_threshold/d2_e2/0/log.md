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
execution time: IAR + RelationalAnalysis = 23.42 + 36.63 = 60.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3248499, upper bound: 0.3248496

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4637
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4637

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248405, upper bound: 0.3223111
time: 8.16 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3223106, upper bound: 0.3248405
time: 6.74 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 14.92 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 14.92
Output dim: 2, lower bound: -0.3248405, upper bound: 0.3223111
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 14.92
Output dim: 2, lower bound: -0.3223106, upper bound: 0.3248405

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6889746, 0.6899805
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5264664, 0.5274160
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5141889, 0.5123847
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6740289, 0.6739421
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5148060, 0.5153217
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6002192, 0.6012695
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6667581, 0.6689720
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6248379, 0.6229753
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4027758, 0.4044124
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6941166, 0.6952808

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 4546

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6224

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246338, upper bound: 0.3223107
time: 5.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248400, upper bound: 0.3221038
time: 6.97 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6899803, 0.6889746
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5274162, 0.5264661
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5123845, 0.5141890
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6739421, 0.6740289
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5153215, 0.5148058
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6012697, 0.6002190
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6689720, 0.6667581
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6229753, 0.6248379
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4044123, 0.4027756
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6952806, 0.6941166

Time for backsubstitution: 23.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4546

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3214543, upper bound: 0.3248385
time: 7.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3223083, upper bound: 0.3239841
time: 6.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 36.88 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.88
Output dim: 2, lower bound: -0.3246338, upper bound: 0.3223107
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.88
Output dim: 2, lower bound: -0.3248400, upper bound: 0.3221038
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.88
Output dim: 2, lower bound: -0.3214543, upper bound: 0.3248385
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 36.88
Output dim: 2, lower bound: -0.3223083, upper bound: 0.3239841

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6843970, 0.6878116
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5226374, 0.5256021
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5136094, 0.5111630
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6733732, 0.6736314
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5106966, 0.5133750
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6001337, 0.6010897
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6628499, 0.6607394
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6188717, 0.6201477
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4003649, 0.3993312
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6853867, 0.6911361

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4546

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4653

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246279, upper bound: 0.3201499
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3224741, upper bound: 0.3223050
time: 5.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6868055, 0.6854029
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5246525, 0.5235870
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5129676, 0.5118053
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6737185, 0.6732862
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5128591, 0.5112123
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6000392, 0.6011841
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6585255, 0.6650641
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6220102, 0.6170092
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3976946, 0.4020015
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6899719, 0.6865506

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 4615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3235977, upper bound: 0.3195210
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222566, upper bound: 0.3208619
time: 3.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6889803, 0.6895745
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5283718, 0.5248737
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5114738, 0.5147338
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6734099, 0.6743467
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5154357, 0.5146127
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6020012, 0.5989981
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6707821, 0.6637380
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6224685, 0.6251388
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4037373, 0.4031776
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6916718, 0.6962786

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6224
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6224

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3212475, upper bound: 0.3248383
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3214538, upper bound: 0.3246318
time: 4.18 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.35
Output dim: 2, lower bound: -0.3246279, upper bound: 0.3201499
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.35
Output dim: 2, lower bound: -0.3224741, upper bound: 0.3223050
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 31.35
Output dim: 2, lower bound: -0.3235977, upper bound: 0.3195210
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 31.35
Output dim: 2, lower bound: -0.3222566, upper bound: 0.3208619
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.35
Output dim: 2, lower bound: -0.3212475, upper bound: 0.3248383
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.35
Output dim: 2, lower bound: -0.3214538, upper bound: 0.3246318

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6843724, 0.6877477
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5215459, 0.5226774
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5123835, 0.5078790
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6718721, 0.6730716
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5019553, 0.5101231
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5997396, 0.6000311
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6610446, 0.6559086
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6174288, 0.6196091
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4002118, 0.3989213
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6753387, 0.6873870

Time for backsubstitution: 22.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246271, upper bound: 0.3194506
time: 5.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3239282, upper bound: 0.3201491
time: 6.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6844018, 0.6874046
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5245423, 0.5230598
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5108943, 0.5135124
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6727533, 0.6740353
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5113268, 0.5126657
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6019156, 0.5988181
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6668744, 0.6555057
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6165023, 0.6223111
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4013267, 0.3980964
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6829414, 0.6921337

Time for backsubstitution: 23.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4650

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3212466, upper bound: 0.3246070
time: 6.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3210156, upper bound: 0.3248366
time: 7.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6868103, 0.6849961
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5265579, 0.5210445
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5102525, 0.5141547
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6730986, 0.6736901
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5134888, 0.5105033
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6018217, 0.5989125
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6625495, 0.6598303
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6196408, 0.6191726
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3986564, 0.4007668
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6875272, 0.6875482

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4650

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3214528, upper bound: 0.3244007
time: 5.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3212220, upper bound: 0.3246303
time: 7.76 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 37.09 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.09
Output dim: 2, lower bound: -0.3246271, upper bound: 0.3194506
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 37.09
Output dim: 2, lower bound: -0.3239282, upper bound: 0.3201491
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 37.09
Output dim: 2, lower bound: -0.3212466, upper bound: 0.3246070
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.09
Output dim: 2, lower bound: -0.3210156, upper bound: 0.3248366
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 37.09
Output dim: 2, lower bound: -0.3214528, upper bound: 0.3244007
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 37.09
Output dim: 2, lower bound: -0.3212220, upper bound: 0.3246303

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6843235, 0.6877086
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5231690, 0.5247648
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5125401, 0.5080016
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6715684, 0.6726639
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5014596, 0.5094661
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5997031, 0.6002123
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6612501, 0.6561975
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6176448, 0.6197646
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4001870, 0.3988903
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6751556, 0.6871421

Time for backsubstitution: 23.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 4650

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3241606, upper bound: 0.3194511
time: 5.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246270, upper bound: 0.3189845
time: 5.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6838150, 0.6865556
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5208263, 0.5180795
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5107820, 0.5133104
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6711097, 0.6728024
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5097485, 0.5105379
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6018920, 0.5988684
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6632490, 0.6528428
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6128712, 0.6195874
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3978772, 0.3955990
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6815085, 0.6910670

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3212457, upper bound: 0.3239072
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3205468, upper bound: 0.3246057
time: 5.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6835527, 0.6868179
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5195622, 0.5193434
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5106924, 0.5133991
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6715207, 0.6723917
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5091987, 0.5110881
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6019654, 0.5987945
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6642113, 0.6518805
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6137786, 0.6186802
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3988276, 0.3946470
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6818738, 0.6907005

Time for backsubstitution: 22.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4653

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3210098, upper bound: 0.3226770
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3188544, upper bound: 0.3248307
time: 5.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6859617, 0.6844091
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5215778, 0.5173280
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5100501, 0.5140414
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6718659, 0.6720464
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5113606, 0.5089254
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6018710, 0.5988886
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6598864, 0.6562052
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6169171, 0.6155417
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3961573, 0.3973173
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6864595, 0.6861150

Time for backsubstitution: 22.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3199787, upper bound: 0.3220475
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3186380, upper bound: 0.3233886
time: 4.08 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 31.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3241606, upper bound: 0.3194511
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3246270, upper bound: 0.3189845
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3212457, upper bound: 0.3239072
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3205468, upper bound: 0.3246057
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3210098, upper bound: 0.3226770
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3188544, upper bound: 0.3248307
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3199787, upper bound: 0.3220475
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 31.53
Output dim: 2, lower bound: -0.3186380, upper bound: 0.3233886

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6843295, 0.6877134
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5231676, 0.5247636
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5125400, 0.5080016
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6715686, 0.6726639
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5014596, 0.5094663
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5997047, 0.6002142
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6612473, 0.6561949
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6176434, 0.6197624
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4001830, 0.3988850
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6751537, 0.6871395

Time for backsubstitution: 22.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4546
type: DSZ, layer: 1, pos: 4650
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 4615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4546

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3237707, upper bound: 0.3189823
time: 4.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3246246, upper bound: 0.3181276
time: 4.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6837757, 0.6865072
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5229135, 0.5197022
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5109046, 0.5134666
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6707025, 0.6724992
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5090919, 0.5100424
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6020725, 0.5988317
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6635385, 0.6530490
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6130266, 0.6198034
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3978462, 0.3955743
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6812639, 0.6908841

Time for backsubstitution: 23.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4615
type: DSZ, layer: 1, pos: 4653
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4615

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3205415, upper bound: 0.3223241
time: 5.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3182654, upper bound: 0.3246010
time: 5.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6834893, 0.6867931
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5166376, 0.5182524
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5074086, 0.5121734
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6709614, 0.6708913
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5059469, 0.5023466
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6009066, 0.5984004
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6593800, 0.6500745
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6132398, 0.6172371
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.3984175, 0.3944937
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6781249, 0.6806536

Time for backsubstitution: 23.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 4615

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3176095, upper bound: 0.3222480
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3162690, upper bound: 0.3235891
time: 4.59 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 32.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 32.78
Output dim: 2, lower bound: -0.3237707, upper bound: 0.3189823
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 32.78
Output dim: 2, lower bound: -0.3246246, upper bound: 0.3181276
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 32.78
Output dim: 2, lower bound: -0.3205415, upper bound: 0.3223241
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 32.78
Output dim: 2, lower bound: -0.3182654, upper bound: 0.3246010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 32.78
Output dim: 2, lower bound: -0.3176095, upper bound: 0.3222480
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 32.78
Output dim: 2, lower bound: -0.3162690, upper bound: 0.3235891

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6843295, 0.6867127
1: -4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5215740, 0.5247636
2: 10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5125400, 0.5070901
3: -3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6715686, 0.6721315
4: -6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5012667, 0.5094663
5: -10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.5984833, 0.6002142
6: -13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6582270, 0.6561949
7: -4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6176434, 0.6192553
8: -4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4001830, 0.3982098
9: -10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6751537, 0.6835299

Time for backsubstitution: 23.19 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 60.05 + 547.69 = 607.73 seconds
