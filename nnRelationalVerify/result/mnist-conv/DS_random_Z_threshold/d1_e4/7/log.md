## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.11948726150000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2823706, 0.2823703)
1: (-12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3489695, 0.3489695)
2: (-9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2716053, 0.2716053)
3: (-0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3234239, 0.3234239)
4: (-11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3780911, 0.3780909)
5: (7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537158, 0.2537158)
6: (-6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2523494, 0.2523494)
7: (-15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4796824, 0.4796824)
8: (-3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417318, 0.2417318)
9: (-3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122460, 0.3122458)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.01 + 34.63 = 58.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.1200877, upper bound: 0.1200876

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 458

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 53

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200345, upper bound: 0.1199320
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199319, upper bound: 0.1200345
time: 4.35 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.03
Output dim: 5, lower bound: -0.1200345, upper bound: 0.1199320
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.03
Output dim: 5, lower bound: -0.1199319, upper bound: 0.1200345

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2823703, 0.2823699
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3489690, 0.3489692
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2716057, 0.2716031
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3234231, 0.3234236
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3780897, 0.3780909
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537160, 0.2537162
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2523493, 0.2523484
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4796820, 0.4796822
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417321, 0.2417318
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122455, 0.3122452

Time for backsubstitution: 22.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 458

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6137

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200342, upper bound: 0.1199025
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200053, upper bound: 0.1199316
time: 5.54 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2823699, 0.2823703
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3489695, 0.3489690
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2716031, 0.2716053
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3234239, 0.3234231
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3780911, 0.3780894
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537158, 0.2537160
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2523484, 0.2523494
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4796822, 0.4796824
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417318, 0.2417318
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122460, 0.3122456

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6137
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6137

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199317, upper bound: 0.1200051
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199026, upper bound: 0.1200343
time: 6.90 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 34.14 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.14
Output dim: 5, lower bound: -0.1200342, upper bound: 0.1199025
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.14
Output dim: 5, lower bound: -0.1200053, upper bound: 0.1199316
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 34.14
Output dim: 5, lower bound: -0.1199317, upper bound: 0.1200051
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 34.14
Output dim: 5, lower bound: -0.1199026, upper bound: 0.1200343

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2825077, 0.2817726
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3491006, 0.3483963
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2717912, 0.2708268
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3238847, 0.3214555
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3781822, 0.3776877
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537258, 0.2536871
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2524066, 0.2520947
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4799552, 0.4785113
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2413669, 0.2418180
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3108624, 0.3125720

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1179526, upper bound: 0.1198690
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200009, upper bound: 0.1178208
time: 3.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2817733, 0.2823699
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3483961, 0.3489692
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2708294, 0.2716031
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3214550, 0.3234236
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3776863, 0.3780909
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2536869, 0.2537162
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2520955, 0.2523484
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4785111, 0.4796822
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417321, 0.2413664
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122455, 0.3108619

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 6232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1179234, upper bound: 0.1198983
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199716, upper bound: 0.1178501
time: 3.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2825072, 0.2817733
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3491008, 0.3483961
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2717886, 0.2708292
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3238854, 0.3214550
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3781836, 0.3776865
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2537258, 0.2536869
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2524058, 0.2520956
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4799552, 0.4785118
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2413664, 0.2418180
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3108624, 0.3125722

Time for backsubstitution: 22.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1188104, upper bound: 0.1200047
time: 5.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199313, upper bound: 0.1188819
time: 4.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2817729, 0.2823703
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3483963, 0.3489690
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2708268, 0.2716053
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3214557, 0.3234231
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3776877, 0.3780894
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2536867, 0.2537160
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2520947, 0.2523494
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4785113, 0.4796824
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417318, 0.2413665
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3122460, 0.3108622

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1187812, upper bound: 0.1200339
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199021, upper bound: 0.1189129
time: 4.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.66 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1179526, upper bound: 0.1198690
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1200009, upper bound: 0.1178208
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1179234, upper bound: 0.1198983
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1199716, upper bound: 0.1178501
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1188104, upper bound: 0.1200047
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1199313, upper bound: 0.1188819
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1187812, upper bound: 0.1200339
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.66
Output dim: 5, lower bound: -0.1199021, upper bound: 0.1189129

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2817211, 0.2808739
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3488412, 0.3481011
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2720101, 0.2710047
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3235683, 0.3211001
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3778342, 0.3772922
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2538729, 0.2538677
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2524505, 0.2521470
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4798865, 0.4784331
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2413391, 0.2417933
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3102971, 0.3119261

Time for backsubstitution: 22.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 6232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1179513, upper bound: 0.1181336
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1162173, upper bound: 0.1198677
time: 4.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2816091, 0.2809860
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3488052, 0.3481369
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2719686, 0.2710457
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3235292, 0.3211392
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3777865, 0.3773398
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2539062, 0.2538344
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2524588, 0.2521386
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4798770, 0.4784429
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2413422, 0.2417902
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3102165, 0.3120070

Time for backsubstitution: 22.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1200009, upper bound: 0.1178172
time: 5.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1198951, upper bound: 0.1178173
time: 3.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2809865, 0.2814711
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3481367, 0.3486743
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2710483, 0.2717810
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3211387, 0.3230685
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3773388, 0.3776951
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2538342, 0.2538966
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2521393, 0.2524008
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4784427, 0.4796040
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417043, 0.2413418
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3116802, 0.3102162

Time for backsubstitution: 23.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1179234, upper bound: 0.1198947
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1178176, upper bound: 0.1198948
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2808745, 0.2815832
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3481009, 0.3487103
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2710068, 0.2718220
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3210996, 0.3231076
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3772906, 0.3777432
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2538676, 0.2538632
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2521478, 0.2523923
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4784329, 0.4796138
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2417076, 0.2413387
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3115996, 0.3102969

Time for backsubstitution: 21.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199716, upper bound: 0.1178465
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1198661, upper bound: 0.1178464
time: 5.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2763011, 0.2746806
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3426404, 0.3410146
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2666755, 0.2663550
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3182697, 0.3146994
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3692906, 0.3699727
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2449110, 0.2459741
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2472115, 0.2461267
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4764695, 0.4744930
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2396771, 0.2406112
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3021617, 0.3026013

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1167288, upper bound: 0.1199713
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1187770, upper bound: 0.1179230
time: 5.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2754147, 0.2755673
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3417194, 0.3419356
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2673147, 0.2657161
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3171295, 0.3158396
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3704698, 0.3687940
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2460129, 0.2448721
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2464367, 0.2469014
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4759367, 0.4750259
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2401596, 0.2401286
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3008916, 0.3038714

Time for backsubstitution: 21.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 481

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 458

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199277, upper bound: 0.1187781
time: 5.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199277, upper bound: 0.1188837
time: 5.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2755668, 0.2752781
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3419359, 0.3415878
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2657137, 0.2671313
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3158400, 0.3166678
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3687952, 0.3703763
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2448721, 0.2460029
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2469004, 0.2463803
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4750254, 0.4756637
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2400422, 0.2401597
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3035450, 0.3008913

Time for backsubstitution: 21.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1187800, upper bound: 0.1182986
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1170459, upper bound: 0.1200326
time: 4.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2746801, 0.2761648
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3410151, 0.3425088
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2663529, 0.2664924
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3146999, 0.3178079
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3699739, 0.3691974
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2459741, 0.2449009
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2461257, 0.2471551
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4744925, 0.4761968
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2405248, 0.2396771
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.3022747, 0.3021613

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 458

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4657

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199002, upper bound: 0.1182037
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1191927, upper bound: 0.1189112
time: 3.97 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 29.76 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1179513, upper bound: 0.1181336
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1162173, upper bound: 0.1198677
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1200009, upper bound: 0.1178172
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1198951, upper bound: 0.1178173
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1179234, upper bound: 0.1198947
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1178176, upper bound: 0.1198948
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1199716, upper bound: 0.1178465
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1198661, upper bound: 0.1178464
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1167288, upper bound: 0.1199713
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1187770, upper bound: 0.1179230
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1199277, upper bound: 0.1187781
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1199277, upper bound: 0.1188837
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1187800, upper bound: 0.1182986
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1170459, upper bound: 0.1200326
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1199002, upper bound: 0.1182037
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 29.76
Output dim: 5, lower bound: -0.1191927, upper bound: 0.1189112

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2779582, 0.2749822
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3469126, 0.3450832
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2717967, 0.2706726
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3155733, 0.3085740
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3709471, 0.3728981
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2484984, 0.2504361
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2505100, 0.2491095
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4735868, 0.4685686
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2361676, 0.2384902
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2959065, 0.3027384

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 6232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4657

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1162155, upper bound: 0.1191582
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1155080, upper bound: 0.1198658
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2732420, 0.2714117
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3349156, 0.3361983
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2719691, 0.2710631
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3225679, 0.3202978
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3627070, 0.3641443
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2532375, 0.2530704
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2460934, 0.2448362
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4700432, 0.4664285
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2399046, 0.2406085
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2918094, 0.2959036

Time for backsubstitution: 21.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4657
type: DSZ, layer: 1, pos: 6232

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1199995, upper bound: 0.1160819
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1182656, upper bound: 0.1178159
time: 3.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2720346, 0.2726190
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3368664, 0.3342474
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2719810, 0.2710457
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3226880, 0.3201779
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3645905, 0.3622608
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2531424, 0.2531645
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2451565, 0.2457727
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4678624, 0.4686086
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2401594, 0.2403527
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2941121, 0.2936001

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6232
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4657

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6232

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.1198937, upper bound: 0.1160820
time: 3.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.1181598, upper bound: 0.1178159
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -11.0019932, -10.2270813, -11.0019932, -10.2270813, -0.2726197, 0.2718968
1: -12.4180117, -11.6260929, -12.4180117, -11.6260929, -0.3342471, 0.3367352
2: -9.6402016, -8.8946152, -9.6402016, -8.8946152, -0.2710483, 0.2717979
3: -0.2500815, 0.5736985, -0.2500815, 0.5736985, -0.3201774, 0.3222271
4: -11.7316227, -10.7969408, -11.7316227, -10.7969408, -0.3622593, 0.3644996
5: 7.6975679, 8.3821516, 7.6975679, 8.3821516, -0.2531652, 0.2531326
6: -6.3868122, -5.5821552, -6.3868122, -5.5821552, -0.2457739, 0.2450982
7: -15.9166489, -14.9498930, -15.9166489, -14.9498930, -0.4686091, 0.4675901
8: -3.8061380, -3.0978208, -3.8061380, -3.0978208, -0.2402668, 0.2401600
9: -3.6112220, -2.9700418, -3.6112220, -2.9700418, -0.2932736, 0.2941124

Time for backsubstitution: 21.41 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.63 + 562.51 = 621.14 seconds
