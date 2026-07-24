## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.23880936800000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4508286, 0.4508287)
1: (-7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4796782, 0.4796782)
2: (2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6314707, 0.6314707)
3: (0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5271082, 0.5271082)
4: (-6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5958090, 0.5958092)
5: (-5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5077269, 0.5077269)
6: (-11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5451224, 0.5451224)
7: (-0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4733357, 0.4733357)
8: (-3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5264854, 0.5264852)
9: (-9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4916582, 0.4916582)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.36 + 32.70 = 55.06 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2595754, upper bound: 0.2595754

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 768

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2572851, upper bound: 0.2565207
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2565207, upper bound: 0.2572851
time: 2.97 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.24
Output dim: 3, lower bound: -0.2572851, upper bound: 0.2565207
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.24
Output dim: 3, lower bound: -0.2565207, upper bound: 0.2572851

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4482727, 0.4465195
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4764886, 0.4743459
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6261168, 0.6320753
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5228581, 0.5236018
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5945826, 0.5963106
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5078659, 0.5067935
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5409439, 0.5405977
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4704213, 0.4730196
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5240488, 0.5254481
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4934847, 0.4850001

Time for backsubstitution: 8.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2499391, upper bound: 0.2478150
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2485826, upper bound: 0.2491716
time: 2.99 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4508286, 0.4482726
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4796782, 0.4764888
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6314707, 0.6261168
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5236018, 0.5271082
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5958090, 0.5945826
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5067935, 0.5077269
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5451224, 0.5409439
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4733357, 0.4704213
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5264854, 0.5240493
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4850001, 0.4916582

Time for backsubstitution: 7.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2491716, upper bound: 0.2485825
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2478150, upper bound: 0.2499391
time: 3.01 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.06 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 3, lower bound: -0.2499391, upper bound: 0.2478150
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 3, lower bound: -0.2485826, upper bound: 0.2491716
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 3, lower bound: -0.2491716, upper bound: 0.2485825
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.06
Output dim: 3, lower bound: -0.2478150, upper bound: 0.2499391

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4472952, 0.4455858
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4743938, 0.4704044
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6245804, 0.6319666
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5241592, 0.5212972
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5895159, 0.5915077
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5058022, 0.5166423
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5417254, 0.5401335
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4699728, 0.4735558
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5229397, 0.5251863
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4875071, 0.4954196

Time for backsubstitution: 7.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 2818

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2462760, upper bound: 0.2331393
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2351460, upper bound: 0.2440837
time: 2.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4482727, 0.4455420
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4725471, 0.4743459
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6260080, 0.6320753
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5205536, 0.5236018
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5897796, 0.5963106
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5078659, 0.5047300
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5404797, 0.5405977
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4704213, 0.4725711
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5240488, 0.5243382
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4934847, 0.4790225

Time for backsubstitution: 7.82 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2818

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2449187, upper bound: 0.2344960
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2337898, upper bound: 0.2454409
time: 3.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4498508, 0.4473387
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4775832, 0.4725473
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6299343, 0.6260080
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5249028, 0.5248041
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5907428, 0.5897796
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5047300, 0.5175757
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5459039, 0.5404797
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4728868, 0.4709575
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5253754, 0.5237877
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4790225, 0.5020778

Time for backsubstitution: 8.48 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2818

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2454409, upper bound: 0.2337898
time: 2.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2344960, upper bound: 0.2449187
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4508286, 0.4472951
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4757366, 0.4764888
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6313620, 0.6261168
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5212972, 0.5271082
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5910065, 0.5945826
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5067935, 0.5056634
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5446584, 0.5409439
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4733357, 0.4699728
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5264854, 0.5229394
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4850001, 0.4856806

Time for backsubstitution: 8.49 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 2818

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2440837, upper bound: 0.2351459
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2331394, upper bound: 0.2462760
time: 2.96 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.52 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2462760, upper bound: 0.2331393
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2351460, upper bound: 0.2440837
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2449187, upper bound: 0.2344960
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2337898, upper bound: 0.2454409
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2454409, upper bound: 0.2337898
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2344960, upper bound: 0.2449187
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2440837, upper bound: 0.2351459
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.52
Output dim: 3, lower bound: -0.2331394, upper bound: 0.2462760

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4471536, 0.4454829
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4726818, 0.4699941
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6121440, 0.6198614
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5107977, 0.5020728
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5629098, 0.5638475
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4920640, 0.4966722
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5087755, 0.4983190
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4406934, 0.4477446
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5207963, 0.5235033
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4792452, 0.4888905

Time for backsubstitution: 8.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2304601, upper bound: 0.2209942
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2334312, upper bound: 0.2206812
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4471922, 0.4454442
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4740133, 0.4686923
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6123013, 0.6195304
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5049345, 0.5075464
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5622022, 0.5649014
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4858322, 0.5027795
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.4999106, 0.5069646
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4440756, 0.4442766
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5212564, 0.5230391
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4809830, 0.4871578

Time for backsubstitution: 8.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2229463, upper bound: 0.2309161
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2231454, upper bound: 0.2273632
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4481311, 0.4454392
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4708350, 0.4739361
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6135707, 0.6199567
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5071919, 0.5044205
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5631733, 0.5685954
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4941061, 0.4847598
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5075109, 0.4987801
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4411595, 0.4467502
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5219064, 0.5226550
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4852259, 0.4724934

Time for backsubstitution: 8.35 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2292756, upper bound: 0.2223564
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2319492, upper bound: 0.2219673
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4481697, 0.4454006
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4721367, 0.4726343
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6137233, 0.6196258
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5013292, 0.5098941
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5621195, 0.5696492
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4878743, 0.4908681
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.4986651, 0.5074259
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4445417, 0.4432917
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5223660, 0.5221908
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4869635, 0.4707607

Time for backsubstitution: 8.87 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2216364, upper bound: 0.2323979
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2217850, upper bound: 0.2286338
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4497008, 0.4472359
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4758704, 0.4721370
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6166859, 0.6137235
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5111520, 0.5055008
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5640850, 0.5621195
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4908681, 0.4976344
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5130312, 0.4986652
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4435778, 0.4450698
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5232263, 0.5221045
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4707606, 0.4958932

Time for backsubstitution: 8.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2286339, upper bound: 0.2217851
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2323976, upper bound: 0.2216364
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4497395, 0.4471973
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4772019, 0.4708352
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6168432, 0.6135709
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5056784, 0.5109744
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5633774, 0.5631733
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4847598, 0.5037415
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5041668, 0.5075110
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4469600, 0.4416783
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5236864, 0.5216446
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4724934, 0.4941604

Time for backsubstitution: 9.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2219672, upper bound: 0.2319497
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2223563, upper bound: 0.2292755
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4506783, 0.4471923
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4740238, 0.4760785
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6181135, 0.6138189
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5075464, 0.5078485
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5643485, 0.5668674
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4929101, 0.4857221
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5117671, 0.4991263
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4440436, 0.4440756
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5243363, 0.5212562
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4767413, 0.4794960

Time for backsubstitution: 9.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2273632, upper bound: 0.2231454
time: 3.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2309156, upper bound: 0.2229463
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4507170, 0.4471537
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4753256, 0.4747767
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6182656, 0.6136668
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5020728, 0.5133221
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5632944, 0.5679212
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4868019, 0.4918303
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5029211, 0.5079724
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4474258, 0.4406934
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5247960, 0.5207963
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4784741, 0.4777634

Time for backsubstitution: 9.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2206811, upper bound: 0.2334317
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2209942, upper bound: 0.2304601
time: 3.22 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 16.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2304601, upper bound: 0.2209942
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2334312, upper bound: 0.2206812
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2229463, upper bound: 0.2309161
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2231454, upper bound: 0.2273632
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2292756, upper bound: 0.2223564
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2319492, upper bound: 0.2219673
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2216364, upper bound: 0.2323979
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2217850, upper bound: 0.2286338
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2286339, upper bound: 0.2217851
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2323976, upper bound: 0.2216364
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2219672, upper bound: 0.2319497
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2223563, upper bound: 0.2292755
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2273632, upper bound: 0.2231454
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2309156, upper bound: 0.2229463
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2206811, upper bound: 0.2334317
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 16.10
Output dim: 3, lower bound: -0.2209942, upper bound: 0.2304601

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 55.06 + 214.31 = 269.38 seconds
