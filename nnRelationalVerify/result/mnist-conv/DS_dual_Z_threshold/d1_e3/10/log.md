## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.20802362000000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7637997, 0.7637992)
1: (-14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4929042, 0.4929042)
2: (-8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5185800, 0.5185800)
3: (-8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5946984, 0.5946989)
4: (-1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5654764, 0.5654764)
5: (-11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6780953, 0.6780949)
6: (-13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4749269, 0.4749269)
7: (-3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3887572, 0.3887572)
8: (-5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6434879, 0.6434879)
9: (4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3716483, 0.3716483)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.20 + 36.15 = 58.35 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.2122686, upper bound: 0.2122684

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 5872

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 567

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2118093, upper bound: 0.2122678
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122679, upper bound: 0.2118092
time: 3.52 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.12 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.12
Output dim: 9, lower bound: -0.2118093, upper bound: 0.2122678
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.12
Output dim: 9, lower bound: -0.2122679, upper bound: 0.2118092

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7612467, 0.7628155
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4925461, 0.4919739
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5184922, 0.5185452
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5946131, 0.5944767
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5651679, 0.5653577
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6776991, 0.6770673
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4738989, 0.4745305
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3887327, 0.3886957
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6433330, 0.6430845
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3710902, 0.3714335

Time for backsubstitution: 20.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5872

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 5872

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2107797, upper bound: 0.2122675
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2118091, upper bound: 0.2112390
time: 3.31 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7628155, 0.7612472
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4919739, 0.4925461
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5185452, 0.5184922
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5944762, 0.5946131
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5653577, 0.5651679
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6770678, 0.6776991
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4745302, 0.4738996
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3886957, 0.3887327
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6430845, 0.6433330
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3714335, 0.3710902

Time for backsubstitution: 21.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5872

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 5872

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2112385, upper bound: 0.2118096
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122677, upper bound: 0.2107801
time: 3.28 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.01 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.01
Output dim: 9, lower bound: -0.2107797, upper bound: 0.2122675
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.01
Output dim: 9, lower bound: -0.2118091, upper bound: 0.2112390
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.01
Output dim: 9, lower bound: -0.2112385, upper bound: 0.2118096
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.01
Output dim: 9, lower bound: -0.2122677, upper bound: 0.2107801

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7582684, 0.7581496
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4872355, 0.4885831
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5178475, 0.5181332
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5890503, 0.5857611
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5606484, 0.5624738
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6693530, 0.6639915
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4687939, 0.4665301
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3804336, 0.3833992
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6410427, 0.6416230
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3687801, 0.3699586

Time for backsubstitution: 21.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2063530, upper bound: 0.2103503
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2088686, upper bound: 0.2078370
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7565813, 0.7598376
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4891558, 0.4866638
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5180802, 0.5179005
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5858974, 0.5889139
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5622845, 0.5608377
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6646228, 0.6687217
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4658985, 0.4694250
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3834360, 0.3803968
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6418710, 0.6407943
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3696158, 0.3691235

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2073788, upper bound: 0.2093267
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2098928, upper bound: 0.2068115
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7598381, 0.7565813
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4866638, 0.4891553
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5179005, 0.5180802
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5889139, 0.5858974
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5608377, 0.5622845
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6687217, 0.6646233
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4694252, 0.4658988
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3803968, 0.3834360
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6407943, 0.6418710
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3691235, 0.3696156

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2068110, upper bound: 0.2098928
time: 5.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2093263, upper bound: 0.2073792
time: 3.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7581491, 0.7582688
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4885836, 0.4872355
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5181332, 0.5178475
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5857611, 0.5890503
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5624738, 0.5606484
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6639915, 0.6693530
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4665298, 0.4687936
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3833992, 0.3804336
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6416230, 0.6410427
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3699586, 0.3687801

Time for backsubstitution: 21.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2078365, upper bound: 0.2088691
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2103503, upper bound: 0.2063535
time: 3.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.62 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2063530, upper bound: 0.2103503
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2088686, upper bound: 0.2078370
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2073788, upper bound: 0.2093267
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2098928, upper bound: 0.2068115
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2068110, upper bound: 0.2098928
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2093263, upper bound: 0.2073792
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2078365, upper bound: 0.2088691
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.62
Output dim: 9, lower bound: -0.2103503, upper bound: 0.2063535

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7543793, 0.7564907
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4809475, 0.4871550
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5151010, 0.5166893
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5861521, 0.5823898
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5598121, 0.5617762
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6607156, 0.6561050
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4649029, 0.4632814
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3778272, 0.3815141
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6392035, 0.6401644
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3674164, 0.3690276

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2060496, upper bound: 0.2095251
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2060760, upper bound: 0.2100497
time: 3.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7566090, 0.7542601
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4858074, 0.4822950
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5164032, 0.5153871
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5856791, 0.5828629
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5599508, 0.5616369
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6614671, 0.6553535
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4655447, 0.4626391
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3785486, 0.3807929
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6395836, 0.6397839
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3678489, 0.3685949

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2085682, upper bound: 0.2075593
time: 4.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2080435, upper bound: 0.2075330
time: 4.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7526913, 0.7581787
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4828672, 0.4852362
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5153337, 0.5164566
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5829992, 0.5855427
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5614481, 0.5601401
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6559854, 0.6608353
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4620075, 0.4661763
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3808296, 0.3785119
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6400318, 0.6393356
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3682520, 0.3681922

Time for backsubstitution: 23.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2070753, upper bound: 0.2085018
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2071018, upper bound: 0.2090258
time: 3.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7549219, 0.7559481
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4877272, 0.4803762
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5166364, 0.5151539
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5825262, 0.5860157
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5615869, 0.5600014
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6567369, 0.6600838
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4626503, 0.4655340
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3815510, 0.3777905
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6404123, 0.6389551
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3686845, 0.3677597

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2095924, upper bound: 0.2065338
time: 5.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2090678, upper bound: 0.2065075
time: 4.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7559481, 0.7549224
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4803762, 0.4877272
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5151539, 0.5166364
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5860157, 0.5825262
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5600014, 0.5615869
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6600842, 0.6567373
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4655342, 0.4626501
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3777905, 0.3815510
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6389551, 0.6404123
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3677597, 0.3686843

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2065076, upper bound: 0.2090677
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2065340, upper bound: 0.2095924
time: 3.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7581787, 0.7526917
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4852362, 0.4828672
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5164566, 0.5153337
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5855427, 0.5829992
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5601406, 0.5614481
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6608357, 0.6559858
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4661760, 0.4620078
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3785117, 0.3808296
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6393356, 0.6400323
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3681922, 0.3682518

Time for backsubstitution: 21.53 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2090259, upper bound: 0.2071016
time: 4.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2085012, upper bound: 0.2070753
time: 5.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7542601, 0.7566094
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4822950, 0.4858074
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5153871, 0.5164032
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5828629, 0.5856791
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5616374, 0.5599508
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6553540, 0.6614671
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4626389, 0.4655449
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3807926, 0.3785486
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6397839, 0.6395841
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3685949, 0.3678489

Time for backsubstitution: 21.64 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2075331, upper bound: 0.2080436
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2075596, upper bound: 0.2085682
time: 3.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7564907, 0.7543793
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4871550, 0.4809475
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5166893, 0.5151010
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5823898, 0.5861521
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5617762, 0.5598116
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6561055, 0.6607156
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4632816, 0.4649026
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3815141, 0.3778272
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6401639, 0.6392035
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3690274, 0.3674164

Time for backsubstitution: 21.56 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 564

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2100499, upper bound: 0.2060758
time: 5.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2095252, upper bound: 0.2060496
time: 4.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2060496, upper bound: 0.2095251
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2060760, upper bound: 0.2100497
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2085682, upper bound: 0.2075593
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2080435, upper bound: 0.2075330
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2070753, upper bound: 0.2085018
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2071018, upper bound: 0.2090258
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2095924, upper bound: 0.2065338
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2090678, upper bound: 0.2065075
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2065076, upper bound: 0.2090677
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2065340, upper bound: 0.2095924
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2090259, upper bound: 0.2071016
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2085012, upper bound: 0.2070753
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2075331, upper bound: 0.2080436
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2075596, upper bound: 0.2085682
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2100499, upper bound: 0.2060758
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.50
Output dim: 9, lower bound: -0.2095252, upper bound: 0.2060496

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7442946, 0.7403970
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4196520, 0.4201257
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.4947863, 0.4931550
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5969853, 0.5886960
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5525169, 0.5575948
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6331244, 0.6289539
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4514680, 0.4490337
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3723705, 0.3770614
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6362529, 0.6359115
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3568032, 0.3576295

Time for backsubstitution: 21.37 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Candidate
type: DSZ, layer: 3, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2001714, upper bound: 0.2062253
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2022140, upper bound: 0.2035377
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7405152, 0.7441754
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4187782, 0.4209995
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.4928689, 0.4950719
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5919857, 0.5936961
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5557694, 0.5543423
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6343155, 0.6277628
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4512973, 0.4492044
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3740959, 0.3753359
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6353316, 0.6368332
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3564508, 0.3579817

Time for backsubstitution: 21.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Candidate
type: DSZ, layer: 3, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2001988, upper bound: 0.2071268
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2020231, upper bound: 0.2036902
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7442946, 0.7403970
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4196520, 0.4201257
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.4947863, 0.4931550
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5969853, 0.5886960
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5525169, 0.5575948
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6331244, 0.6289539
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4514680, 0.4490337
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3723705, 0.3770614
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6362529, 0.6359115
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3568032, 0.3576295

Time for backsubstitution: 21.74 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Candidate
type: DSZ, layer: 3, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2022081, upper bound: 0.2035066
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2056450, upper bound: 0.2016842
time: 4.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7405152, 0.7441754
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4187782, 0.4209995
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.4928689, 0.4950719
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5919857, 0.5936961
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5557694, 0.5543423
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6343155, 0.6277628
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4512973, 0.4492044
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3740959, 0.3753359
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6353316, 0.6368332
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3564508, 0.3579817

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Candidate
type: DSZ, layer: 3, pos: 711

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2020561, upper bound: 0.2036989
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2047432, upper bound: 0.2016579
time: 4.05 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2001714, upper bound: 0.2062253
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2022140, upper bound: 0.2035377
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2001988, upper bound: 0.2071268
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2020231, upper bound: 0.2036902
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2022081, upper bound: 0.2035066
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2056450, upper bound: 0.2016842
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2020561, upper bound: 0.2036989
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.69
Output dim: 9, lower bound: -0.2047432, upper bound: 0.2016579
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2070753, upper bound: 0.2085018
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2071018, upper bound: 0.2090258
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2095924, upper bound: 0.2065338
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2090678, upper bound: 0.2065075
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2065076, upper bound: 0.2090677
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2065340, upper bound: 0.2095924
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2090259, upper bound: 0.2071016
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2085012, upper bound: 0.2070753
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2075331, upper bound: 0.2080436
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2075596, upper bound: 0.2085682
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2100499, upper bound: 0.2060758
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.69
Output dim: 9, lower bound: -0.2095252, upper bound: 0.2060496

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.35 + 548.49 = 606.84 seconds
