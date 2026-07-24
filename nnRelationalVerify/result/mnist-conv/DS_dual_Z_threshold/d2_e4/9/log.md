## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6361788214999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7538667, 1.7538671)
1: (-16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5473294, 1.5473294)
2: (-7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5473089, 1.5473089)
3: (-12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.6015015, 2.6015015)
4: (-3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7983255, 1.7983255)
5: (-13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.2112970, 1.2112970)
6: (-15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7188849, 1.7188859)
7: (-7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1962242, 2.1962242)
8: (-6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5854363, 1.5854363)
9: (4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4497461, 1.4497461)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.67 + 34.11 = 56.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.6393757, upper bound: 0.6393755

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4557
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4557

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393564, upper bound: 0.6324031
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6324034, upper bound: 0.6393561
time: 4.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.71 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.71
Output dim: 9, lower bound: -0.6393564, upper bound: 0.6324031
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.71
Output dim: 9, lower bound: -0.6324034, upper bound: 0.6393561

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7523375, 1.7528358
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5462027, 1.5463524
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5316286, 1.5335836
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5918417, 2.5930462
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7642145, 1.7709084
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1914854, 1.1886530
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7211514, 1.7214718
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1692533, 2.1595526
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5613656, 1.5579362
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4333711, 1.4310298

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 906

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393561, upper bound: 0.6311723
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6381412, upper bound: 0.6324028
time: 4.62 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7528353, 1.7523370
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5463524, 1.5462027
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5335836, 1.5316286
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5930462, 2.5918417
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7709084, 1.7642140
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1886530, 1.1914854
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7214718, 1.7211514
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1595526, 2.1692543
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5579362, 1.5613656
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4310298, 1.4333711

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 906
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 906

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6324031, upper bound: 0.6381408
time: 5.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6311726, upper bound: 0.6393557
time: 4.50 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.71 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 9, lower bound: -0.6393561, upper bound: 0.6311723
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 9, lower bound: -0.6381412, upper bound: 0.6324028
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 9, lower bound: -0.6324031, upper bound: 0.6381408
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.71
Output dim: 9, lower bound: -0.6311726, upper bound: 0.6393557

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7506213, 1.7508750
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5471740, 1.5476527
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5336714, 1.5361705
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5970249, 2.5975122
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7564874, 1.7620764
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1841860, 1.1803119
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7211475, 1.7217369
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1629477, 2.1523466
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5577106, 1.5555501
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4332662, 1.4309096

Time for backsubstitution: 22.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393553, upper bound: 0.6309207
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6391045, upper bound: 0.6311709
time: 4.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7503772, 1.7511196
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5475030, 1.5473232
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5342155, 1.5356269
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5963087, 2.5982285
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7553821, 1.7631817
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1831446, 1.1813533
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7214165, 1.7214684
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1620474, 2.1532459
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5589795, 1.5542812
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4332514, 1.4309249

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6381399, upper bound: 0.6321512
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6378895, upper bound: 0.6324014
time: 4.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7511191, 1.7503767
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5473232, 1.5475030
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5356269, 1.5342155
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5982285, 2.5963087
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7631822, 1.7553825
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1813536, 1.1831443
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7214689, 1.7214165
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1532459, 2.1620474
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5542812, 1.5589795
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4309249, 1.4332509

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6324018, upper bound: 0.6378891
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6321515, upper bound: 0.6381394
time: 5.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7508750, 1.7506208
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5476527, 1.5471735
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5361705, 1.5336714
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5975122, 2.5970249
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7620759, 1.7564878
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1803122, 1.1841857
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7217369, 1.7211480
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1523466, 2.1629477
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5555501, 1.5577106
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4309101, 1.4332662

Time for backsubstitution: 22.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4610
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 4610

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6311714, upper bound: 0.6391041
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6309213, upper bound: 0.6393543
time: 4.34 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.69 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6393553, upper bound: 0.6309207
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6391045, upper bound: 0.6311709
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6381399, upper bound: 0.6321512
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6378895, upper bound: 0.6324014
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6324018, upper bound: 0.6378891
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6321515, upper bound: 0.6381394
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6311714, upper bound: 0.6391041
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.69
Output dim: 9, lower bound: -0.6309213, upper bound: 0.6393543

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7448683, 1.7428284
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5406828, 1.5385752
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5334878, 1.5359135
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5901499, 2.5926018
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7558937, 1.7612524
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1783075, 1.1720917
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7204685, 1.7207870
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1621532, 2.1512327
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5563154, 1.5535965
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4326062, 1.4299870

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6387247, upper bound: 0.6309191
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393529, upper bound: 0.6302925
time: 4.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7425737, 1.7451229
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5380964, 1.5411615
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5334148, 1.5359869
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5921144, 2.5906372
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7556639, 1.7614827
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1759663, 1.1744335
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7201977, 1.7210579
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1618338, 2.1515522
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5557570, 1.5541549
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4323430, 1.4302497

Time for backsubstitution: 22.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6384744, upper bound: 0.6311691
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6391025, upper bound: 0.6305428
time: 4.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7446241, 1.7430730
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5410123, 1.5382457
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5340314, 1.5353699
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5894337, 2.5933189
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7547884, 1.7623577
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1772661, 1.1731331
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7207375, 1.7205181
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1612530, 2.1521330
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5575843, 1.5523276
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4325910, 1.4300017

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6375098, upper bound: 0.6321493
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6381379, upper bound: 0.6315216
time: 4.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7423296, 1.7453671
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5384259, 1.5408320
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5339584, 1.5354428
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5913982, 2.5913544
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7545586, 1.7625880
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1749249, 1.1754749
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7204666, 1.7207890
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1609344, 2.1524515
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5570259, 1.5528860
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4323277, 1.4302645

Time for backsubstitution: 22.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6372594, upper bound: 0.6323996
time: 4.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6378875, upper bound: 0.6317720
time: 4.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7453680, 1.7423296
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5408320, 1.5384259
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5354428, 1.5339584
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5913544, 2.5913982
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7625885, 1.7545586
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1754751, 1.1749241
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7207890, 1.7204666
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1524515, 2.1609344
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5528860, 1.5570259
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4302649, 1.4323282

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6317721, upper bound: 0.6378876
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6323998, upper bound: 0.6372609
time: 4.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7430735, 1.7446241
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5382457, 1.5410123
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5353699, 1.5340314
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5933189, 2.5894337
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7623577, 1.7547884
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1731339, 1.1772659
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7205181, 1.7207375
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1521330, 2.1612530
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5523276, 1.5575843
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4300017, 1.4325910

Time for backsubstitution: 22.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6315218, upper bound: 0.6381394
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6321495, upper bound: 0.6375113
time: 4.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7451220, 1.7425742
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5411615, 1.5380964
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5359864, 1.5334148
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5906372, 2.5921144
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7614822, 1.7556639
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1744337, 1.1759655
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7210579, 1.7201977
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1515522, 2.1618338
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5541549, 1.5557570
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4302497, 1.4323430

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6305430, upper bound: 0.6391019
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6311693, upper bound: 0.6384753
time: 4.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7428293, 1.7448683
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5385752, 1.5406828
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5359139, 1.5334878
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5926018, 2.5901508
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7612524, 1.7558942
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1720915, 1.1783078
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7207870, 1.7204685
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1512327, 2.1621532
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5535965, 1.5563154
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4299870, 1.4326057

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4608
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 4608

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6302929, upper bound: 0.6393523
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6309193, upper bound: 0.6387256
time: 4.34 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6387247, upper bound: 0.6309191
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6393529, upper bound: 0.6302925
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6384744, upper bound: 0.6311691
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6391025, upper bound: 0.6305428
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6375098, upper bound: 0.6321493
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6381379, upper bound: 0.6315216
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6372594, upper bound: 0.6323996
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6378875, upper bound: 0.6317720
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6317721, upper bound: 0.6378876
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6323998, upper bound: 0.6372609
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6315218, upper bound: 0.6381394
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6321495, upper bound: 0.6375113
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6305430, upper bound: 0.6391019
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6311693, upper bound: 0.6384753
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6302929, upper bound: 0.6393523
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.26
Output dim: 9, lower bound: -0.6309193, upper bound: 0.6387256

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7437325, 1.7427459
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5405507, 1.5367656
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5333595, 1.5341854
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5898943, 2.5925817
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7556472, 1.7579436
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1768932, 1.1719825
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7195101, 1.7207127
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1619682, 2.1512194
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5562387, 1.5525122
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4318585, 1.4299307

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6380056, upper bound: 0.6309203
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6387241, upper bound: 0.6302005
time: 4.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7447863, 1.7416921
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5388732, 1.5384436
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5317593, 1.5357852
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5901308, 2.5923462
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7525859, 1.7610049
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1781979, 1.1706774
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7203951, 1.7198281
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1621399, 2.1510487
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5552306, 1.5535192
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4325504, 1.4292393

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6386333, upper bound: 0.6302938
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6393524, upper bound: 0.6295738
time: 4.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.9221611, -5.9657760, -8.9221611, -5.9657760, -1.7414379, 1.7450404
1: -16.5001736, -14.1283016, -16.5001736, -14.1283016, -1.5379643, 1.5393519
2: -7.4758034, -4.9801350, -7.4758034, -4.9801350, -1.5332861, 1.5342584
3: -12.9350147, -9.6876116, -12.9350147, -9.6876116, -2.5918589, 2.5906172
4: -3.2633729, -0.9272225, -3.2633729, -0.9272225, -1.7554164, 1.7581739
5: -13.3927860, -10.7282457, -13.3927860, -10.7282457, -1.1745520, 1.1743243
6: -15.2848930, -12.3271980, -15.2848930, -12.3271980, -1.7192392, 1.7209835
7: -7.6543331, -5.0435581, -7.6543331, -5.0435581, -2.1616497, 2.1515388
8: -6.0455451, -3.6904855, -6.0455451, -3.6904855, -1.5556798, 1.5530701
9: 4.5871305, 6.2297759, 4.5871305, 6.2297759, -1.4315953, 1.4301939

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 5875
type: DSZ, layer: 1, pos: 5798
type: DSZ, layer: 1, pos: 961

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 884

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6377552, upper bound: 0.6311703
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.6384738, upper bound: 0.6304501
time: 4.26 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 31.04 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.04
Output dim: 9, lower bound: -0.6380056, upper bound: 0.6309203
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.04
Output dim: 9, lower bound: -0.6387241, upper bound: 0.6302005
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.04
Output dim: 9, lower bound: -0.6386333, upper bound: 0.6302938
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.04
Output dim: 9, lower bound: -0.6393524, upper bound: 0.6295738
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.04
Output dim: 9, lower bound: -0.6377552, upper bound: 0.6311703
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.04
Output dim: 9, lower bound: -0.6384738, upper bound: 0.6304501
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6391025, upper bound: 0.6305428
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6375098, upper bound: 0.6321493
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6381379, upper bound: 0.6315216
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6372594, upper bound: 0.6323996
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6378875, upper bound: 0.6317720
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6317721, upper bound: 0.6378876
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6323998, upper bound: 0.6372609
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6315218, upper bound: 0.6381394
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6321495, upper bound: 0.6375113
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6305430, upper bound: 0.6391019
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6311693, upper bound: 0.6384753
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6302929, upper bound: 0.6393523
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.04
Output dim: 9, lower bound: -0.6309193, upper bound: 0.6387256

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.77 + 547.86 = 604.64 seconds
