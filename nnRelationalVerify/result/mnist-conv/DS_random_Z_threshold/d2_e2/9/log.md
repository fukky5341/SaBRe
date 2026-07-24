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
execution time: IAR + RelationalAnalysis = 24.44 + 32.91 = 57.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2595754, upper bound: 0.2595754

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1103
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 2534
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1438

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1103

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2436788, upper bound: 0.2463281
time: 2.89 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2463280, upper bound: 0.2436788
time: 2.88 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.78 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.78
Output dim: 3, lower bound: -0.2436788, upper bound: 0.2463281
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.78
Output dim: 3, lower bound: -0.2463280, upper bound: 0.2436788

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4416900, 0.4348745
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4316626, 0.4170387
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6190777, 0.6225042
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5053825, 0.5127370
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5978894, 0.5981987
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4843256, 0.4922853
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5259101, 0.5333006
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4552646, 0.4552212
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4520485, 0.4439178
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4704113, 0.4615941

Time for backsubstitution: 8.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2534
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1511

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 768

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2418680, upper bound: 0.2432657
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2405042, upper bound: 0.2442294
time: 2.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4348745, 0.4416898
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4170387, 0.4316626
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6225042, 0.6190777
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5127370, 0.5053825
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5981984, 0.5978892
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4922853, 0.4843256
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5333006, 0.5259099
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4552212, 0.4552646
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4439178, 0.4520485
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4615941, 0.4704114

Time for backsubstitution: 8.76 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2534
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2534

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2460410, upper bound: 0.2433704
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2460198, upper bound: 0.2433916
time: 2.98 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.79
Output dim: 3, lower bound: -0.2418680, upper bound: 0.2432657
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.79
Output dim: 3, lower bound: -0.2405042, upper bound: 0.2442294
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.79
Output dim: 3, lower bound: -0.2460410, upper bound: 0.2433704
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.79
Output dim: 3, lower bound: -0.2460198, upper bound: 0.2433916

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4385321, 0.4299638
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4282918, 0.4115252
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6143165, 0.6237001
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5020790, 0.5093415
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5966620, 0.5986984
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4842749, 0.4911621
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5219913, 0.5293202
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4523642, 0.4547791
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4496367, 0.4429045
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4728292, 0.4555094

Time for backsubstitution: 8.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 2534
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 963

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2410870, upper bound: 0.2419609
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2414313, upper bound: 0.2429572
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4416900, 0.4317168
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4316626, 0.4136679
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6190777, 0.6177430
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5019870, 0.5127370
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5978894, 0.5969713
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4832025, 0.4922853
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5259101, 0.5293820
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4552646, 0.4523208
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4520485, 0.4415057
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4643267, 0.4615941

Time for backsubstitution: 9.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2534
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 327

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 222

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2369738, upper bound: 0.2384807
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2369738, upper bound: 0.2386326
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4348745, 0.4416909
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4170387, 0.4316641
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6225042, 0.6190777
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5127385, 0.5053825
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5981967, 0.5978885
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4922848, 0.4843228
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5333004, 0.5259073
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4552212, 0.4552641
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4439194, 0.4520483
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4615939, 0.4704115

Time for backsubstitution: 8.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1511

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2291250, upper bound: 0.2285725
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2289860, upper bound: 0.2285724
time: 2.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4348745, 0.4416898
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4170387, 0.4316626
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6225042, 0.6190777
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5127370, 0.5053825
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5981982, 0.5978892
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4922853, 0.4843252
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5333006, 0.5259099
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4552212, 0.4552646
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4439173, 0.4520485
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4615941, 0.4704113

Time for backsubstitution: 9.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1511

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2449122, upper bound: 0.2426501
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2453001, upper bound: 0.2423279
time: 2.81 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 14.98 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2410870, upper bound: 0.2419609
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2414313, upper bound: 0.2429572
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2369738, upper bound: 0.2384807
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2369738, upper bound: 0.2386326
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2291250, upper bound: 0.2285725
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2289860, upper bound: 0.2285724
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2449122, upper bound: 0.2426501
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 3, lower bound: -0.2453001, upper bound: 0.2423279

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4382246, 0.4298389
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4290249, 0.4124506
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6153965, 0.6248078
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5013971, 0.5085227
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5968101, 0.5989089
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4841180, 0.4909875
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5218034, 0.5290949
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4523430, 0.4547334
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4494891, 0.4434471
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4727172, 0.4555036

Time for backsubstitution: 8.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2534
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 3124

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2340013, upper bound: 0.2333649
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2327706, upper bound: 0.2348287
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4384074, 0.4296575
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4292424, 0.4122584
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6154242, 0.6247811
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5012603, 0.5088596
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5968983, 0.5988467
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4841001, 0.4910200
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5218196, 0.5291324
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4523349, 0.4547579
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4501793, 0.4429860
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4728402, 0.4553975

Time for backsubstitution: 9.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 2534
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1511
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 2131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1438

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2344708, upper bound: 0.2343626
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2333059, upper bound: 0.2358263
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4348722, 0.4416778
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4170372, 0.4316587
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6225014, 0.6190801
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5127492, 0.5053775
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5981925, 0.5978842
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4922838, 0.4843247
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5332985, 0.5259106
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4552188, 0.4552708
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4439030, 0.4520433
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4615972, 0.4704108

Time for backsubstitution: 9.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 1438

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2279545, upper bound: 0.2278320
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2278156, upper bound: 0.2278319
time: 2.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4348745, 0.4416873
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4170387, 0.4316611
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6225042, 0.6190748
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5127320, 0.5053825
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5981982, 0.5978837
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4922853, 0.4843242
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5333006, 0.5259075
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4552212, 0.4552624
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4439123, 0.4520485
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4615934, 0.4704113

Time for backsubstitution: 9.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2818

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2416288, upper bound: 0.2306144
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2313720, upper bound: 0.2379612
time: 2.91 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 15.13 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2340013, upper bound: 0.2333649
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2327706, upper bound: 0.2348287
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2344708, upper bound: 0.2343626
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2333059, upper bound: 0.2358263
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2279545, upper bound: 0.2278320
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2278156, upper bound: 0.2278319
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2416288, upper bound: 0.2306144
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 15.13
Output dim: 3, lower bound: -0.2313720, upper bound: 0.2379612

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4348843, 0.4417359
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4203707, 0.4359922
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.5995140, 0.5962381
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.4840162, 0.4711936
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5647571, 0.5633886
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4805911, 0.4649665
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.4955769, 0.4798708
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4039352, 0.4073584
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4389799, 0.4475753
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4545698, 0.4651204

Time for backsubstitution: 8.73 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3118
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 2899

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3118

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2412555, upper bound: 0.2226411
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2345423, upper bound: 0.2302829
time: 3.13 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.67 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.67
Output dim: 3, lower bound: -0.2412555, upper bound: 0.2226411
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 14.67
Output dim: 3, lower bound: -0.2345423, upper bound: 0.2302829

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.3997836, 0.4062397
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4228796, 0.4429748
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.5556717, 0.5542006
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.4851527, 0.4711130
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5705767, 0.5699375
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4766881, 0.4610367
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.4592340, 0.4435303
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.3762002, 0.3762964
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4412587, 0.4506900
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4130859, 0.4260494

Time for backsubstitution: 9.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1485
type: DSZ, layer: 3, pos: 2899
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 899

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1485

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2378016, upper bound: 0.2212578
time: 2.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2398721, upper bound: 0.2191874
time: 3.01 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 15.12 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 15.12
Output dim: 3, lower bound: -0.2378016, upper bound: 0.2212578
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.12
Output dim: 3, lower bound: -0.2398721, upper bound: 0.2191874

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.3942269, 0.4009722
1: -7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4215462, 0.4423910
2: 2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.5477924, 0.5418630
3: 0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.4780252, 0.4631634
4: -6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5610821, 0.5613952
5: -5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.4633975, 0.4482143
6: -11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.4476042, 0.4328864
7: -0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.3599136, 0.3599470
8: -3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.4208035, 0.4302478
9: -9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4011171, 0.4149643

Time for backsubstitution: 8.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 181
type: DSZ, layer: 3, pos: 221
type: DSZ, layer: 3, pos: 1754
type: DSZ, layer: 3, pos: 1438
type: DSZ, layer: 3, pos: 222
type: DSZ, layer: 3, pos: 899
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 768
type: DSZ, layer: 3, pos: 922
type: DSZ, layer: 3, pos: 327
type: DSZ, layer: 3, pos: 66
type: DSZ, layer: 3, pos: 963
type: DSZ, layer: 3, pos: 183
type: DSZ, layer: 3, pos: 3124
type: DSZ, layer: 3, pos: 1706
type: DSZ, layer: 3, pos: 2899

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2225271, upper bound: 0.2054436
time: 3.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2224845, upper bound: 0.2054433
time: 3.90 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 21.27 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 21.27
Output dim: 3, lower bound: -0.2225271, upper bound: 0.2054436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 21.27
Output dim: 3, lower bound: -0.2224845, upper bound: 0.2054433

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 57.35 + 205.62 = 262.98 seconds
