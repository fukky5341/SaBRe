## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.2745331572


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.6024015, 0.6024015)
1: (-7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5952206, 0.5952203)
2: (-2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5117292, 0.5117292)
3: (5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5401940, 0.5401940)
4: (-11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5841854, 0.5841851)
5: (-2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5143909, 0.5143909)
6: (-9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6629021, 0.6629021)
7: (-7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6346526, 0.6346531)
8: (-2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5288428, 0.5288427)
9: (-4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4562324, 0.4562323)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.67 + 34.42 = 57.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2756352, upper bound: 0.2756360

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5842
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 5842

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756089, upper bound: 0.2756216
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2756216, upper bound: 0.2756096
time: 3.42 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.80 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.80
Output dim: 3, lower bound: -0.2756089, upper bound: 0.2756216
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.80
Output dim: 3, lower bound: -0.2756216, upper bound: 0.2756096

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5940909, 0.5954759
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5952201, 0.5953791
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5076125, 0.5088712
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5456469, 0.5444574
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5841839, 0.5847499
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5085357, 0.5073507
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6584880, 0.6596036
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6191761, 0.6217551
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5267426, 0.5273194
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4579604, 0.4582694

Time for backsubstitution: 20.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 928

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755661, upper bound: 0.2755650
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2755642
time: 3.82 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5954757, 0.5940909
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5953791, 0.5952199
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5088713, 0.5076126
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5444574, 0.5456468
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5847499, 0.5841839
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5073508, 0.5085359
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6596038, 0.6584878
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6217549, 0.6191761
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5273193, 0.5267427
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4582694, 0.4579602

Time for backsubstitution: 21.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 928

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2755642
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2755662
time: 3.62 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.71 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2755661, upper bound: 0.2755650
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2755642
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2755642
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.71
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2755662

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5910020, 0.5935271
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5937192, 0.5930018
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5063996, 0.5069470
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5461953, 0.5444534
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5841658, 0.5858712
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5074701, 0.5056633
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6539614, 0.6575019
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6193962, 0.6217542
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5159794, 0.5205282
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4568402, 0.4564941

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755651, upper bound: 0.2742001
time: 5.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742023, upper bound: 0.2755640
time: 3.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5921419, 0.5923867
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5928428, 0.5935585
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5056884, 0.5062397
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5456426, 0.5450058
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5853052, 0.5847318
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5068483, 0.5062853
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6563861, 0.6550772
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6191752, 0.6215298
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5199517, 0.5165561
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4561852, 0.4565246

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755632, upper bound: 0.2742004
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742004, upper bound: 0.2755639
time: 4.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5923867, 0.5921419
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5935585, 0.5928426
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5062397, 0.5056884
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5450056, 0.5456427
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5847318, 0.5853050
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5062852, 0.5068485
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6550772, 0.6563861
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6215298, 0.6191752
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5165561, 0.5199517
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4565247, 0.4561851

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755632, upper bound: 0.2742002
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742004, upper bound: 0.2755640
time: 4.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5935271, 0.5910020
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5930018, 0.5937190
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5069470, 0.5063996
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5444534, 0.5461955
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5858712, 0.5841656
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5056634, 0.5074704
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6575019, 0.6539614
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6217539, 0.6193964
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5205282, 0.5159794
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4564942, 0.4568402

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5847
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 5847

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755632, upper bound: 0.2742021
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2742004, upper bound: 0.2755658
time: 4.26 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.40 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2755651, upper bound: 0.2742001
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2742023, upper bound: 0.2755640
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2755632, upper bound: 0.2742004
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2742004, upper bound: 0.2755639
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2755632, upper bound: 0.2742002
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2742004, upper bound: 0.2755640
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2755632, upper bound: 0.2742021
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.40
Output dim: 3, lower bound: -0.2742004, upper bound: 0.2755658

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5847931, 0.5894163
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5853124, 0.5803089
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5062079, 0.5066603
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5434828, 0.5403590
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5817599, 0.5842772
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5051039, 0.5020912
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6520212, 0.6545708
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6127923, 0.6173794
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5143998, 0.5181433
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4512640, 0.4528024

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2728790
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742487, upper bound: 0.2741977
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5868912, 0.5873184
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5810263, 0.5845950
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5061131, 0.5067554
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5421011, 0.5417407
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5825717, 0.5834653
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5038983, 0.5032969
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6510303, 0.6555617
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6150217, 0.6151500
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5135946, 0.5189484
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4531485, 0.4509181

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742014, upper bound: 0.2742427
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2728859, upper bound: 0.2755606
time: 3.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5859332, 0.5882761
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5844359, 0.5808656
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5054965, 0.5059531
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5429301, 0.5409114
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5828993, 0.5831378
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5044819, 0.5027131
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6544459, 0.6521461
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6125710, 0.6171553
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5183721, 0.5141714
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4506091, 0.4528329

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755599, upper bound: 0.2728791
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742419, upper bound: 0.2741974
time: 3.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5880313, 0.5861781
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5801499, 0.5851517
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5054016, 0.5060481
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5415485, 0.5422931
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5837111, 0.5823259
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5032762, 0.5039188
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6534550, 0.6531367
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6148007, 0.6149259
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5175669, 0.5149764
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4524935, 0.4509486

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2741971, upper bound: 0.2742419
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2728791, upper bound: 0.2755598
time: 6.82 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5861781, 0.5880313
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5851517, 0.5801499
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5060482, 0.5054017
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5422931, 0.5415485
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5823259, 0.5837111
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5039188, 0.5032762
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6531370, 0.6534550
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6149259, 0.6148007
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5149765, 0.5175668
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4509486, 0.4524934

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755598, upper bound: 0.2728790
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742420, upper bound: 0.2741975
time: 3.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5882761, 0.5859332
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5808659, 0.5844357
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5059528, 0.5054967
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5409114, 0.5429300
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5831378, 0.5828991
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5027131, 0.5044819
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6521461, 0.6544459
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6171553, 0.6125710
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5141714, 0.5183719
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4528331, 0.4506091

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2741970, upper bound: 0.2742419
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2728792, upper bound: 0.2755598
time: 5.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5873182, 0.5868912
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5845950, 0.5810263
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5067554, 0.5061131
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5417407, 0.5421011
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5834653, 0.5825717
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5032970, 0.5038981
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6555617, 0.6510303
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6151500, 0.6150217
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5189486, 0.5135946
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4509181, 0.4531485

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2755598, upper bound: 0.2728858
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2742420, upper bound: 0.2742020
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5894163, 0.5847931
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5803092, 0.5853124
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5066605, 0.5062081
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5403590, 0.5434828
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5842772, 0.5817597
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5020913, 0.5051038
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6545708, 0.6520209
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6173794, 0.6127923
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5181434, 0.5143996
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4528025, 0.4512640

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 942

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2741970, upper bound: 0.2742486
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2728792, upper bound: 0.2755649
time: 4.73 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2755642, upper bound: 0.2728790
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2742487, upper bound: 0.2741977
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2742014, upper bound: 0.2742427
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2728859, upper bound: 0.2755606
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2755599, upper bound: 0.2728791
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2742419, upper bound: 0.2741974
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2741971, upper bound: 0.2742419
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2728791, upper bound: 0.2755598
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2755598, upper bound: 0.2728790
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2742420, upper bound: 0.2741975
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2741970, upper bound: 0.2742419
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2728792, upper bound: 0.2755598
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2755598, upper bound: 0.2728858
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2742420, upper bound: 0.2742020
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2741970, upper bound: 0.2742486
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.64
Output dim: 3, lower bound: -0.2728792, upper bound: 0.2755649

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5868180, 0.5911279
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5802696, 0.5766575
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5030978, 0.5041643
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5458095, 0.5423245
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5791557, 0.5828331
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5048363, 0.5017172
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6530344, 0.6557705
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6075885, 0.6130435
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5113759, 0.5158597
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4527254, 0.4550899

Time for backsubstitution: 21.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 933

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2744462, upper bound: 0.2728393
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2741623, upper bound: 0.2728401
time: 3.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5886028, 0.5893431
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5773761, 0.5795522
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5036175, 0.5036451
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5440667, 0.5440679
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5811288, 0.5808613
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5035241, 0.5030293
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6522300, 0.6565750
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6106861, 0.6099465
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5113111, 0.5159246
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4554365, 0.4523795

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 933

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2728462, upper bound: 0.2741577
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2728454, upper bound: 0.2744422
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5879581, 0.5899873
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5793931, 0.5772154
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5023863, 0.5034570
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5452569, 0.5428770
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5802951, 0.5816941
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5042145, 0.5023391
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6554592, 0.6533458
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6073673, 0.6128194
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5153480, 0.5118877
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4520705, 0.4551212

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 933

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2744419, upper bound: 0.2728393
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2741580, upper bound: 0.2728402
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9493103, -7.8348112, -8.9493103, -7.8348112, -0.5897424, 0.5882025
1: -7.1685023, -6.0953550, -7.1685023, -6.0953550, -0.5764995, 0.5801089
2: -2.9241743, -1.8778169, -2.9241743, -1.8778169, -0.5029061, 0.5029377
3: 5.8405285, 6.8184710, 5.8405285, 6.8184710, -0.5435140, 0.5446200
4: -11.6864643, -10.4383507, -11.6864643, -10.4383507, -0.5822670, 0.5797219
5: -2.0644875, -1.1097507, -2.0644875, -1.1097507, -0.5029023, 0.5036513
6: -9.6060925, -8.3918819, -9.6060925, -8.3918819, -0.6546550, 0.6541502
7: -7.1219349, -6.0866470, -7.1219349, -6.0866470, -0.6104648, 0.6097221
8: -2.1370850, -1.2078891, -2.1370850, -1.2078891, -0.5152831, 0.5119525
9: -4.3173146, -3.3542800, -4.3173146, -3.3542800, -0.4547815, 0.4524099

Time for backsubstitution: 21.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 4656
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 933

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2728394, upper bound: 0.2741586
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2728386, upper bound: 0.2744425
time: 4.04 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2744462, upper bound: 0.2728393
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2741623, upper bound: 0.2728401
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2728462, upper bound: 0.2741577
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2728454, upper bound: 0.2744422
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2744419, upper bound: 0.2728393
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2741580, upper bound: 0.2728402
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2728394, upper bound: 0.2741586
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.37
Output dim: 3, lower bound: -0.2728386, upper bound: 0.2744425
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.37
Output dim: 3, lower bound: -0.2755598, upper bound: 0.2728790
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.37
Output dim: 3, lower bound: -0.2728792, upper bound: 0.2755598
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.37
Output dim: 3, lower bound: -0.2755598, upper bound: 0.2728858
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.37
Output dim: 3, lower bound: -0.2728792, upper bound: 0.2755649

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.09 + 550.57 = 607.66 seconds
