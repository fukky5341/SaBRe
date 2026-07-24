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
execution time: IAR + RelationalAnalysis = 23.83 + 35.32 = 59.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.2122686, upper bound: 0.2122684

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5872
type: DSZ, layer: 1, pos: 567

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5872

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2112390, upper bound: 0.2122688
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122684, upper bound: 0.2112394
time: 3.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.88
Output dim: 9, lower bound: -0.2112390, upper bound: 0.2122688
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.88
Output dim: 9, lower bound: -0.2122684, upper bound: 0.2112394

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7608204, 0.7591333
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4875941, 0.4895129
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5179362, 0.5181689
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5891356, 0.5859828
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5609560, 0.5625920
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6697493, 0.6650190
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4698215, 0.4669266
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3804584, 0.3834608
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6411977, 0.6420259
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3693376, 0.3701732

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 567

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 567

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2107797, upper bound: 0.2122675
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2112385, upper bound: 0.2118096
time: 3.10 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7591343, 0.7608209
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4895129, 0.4875941
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5181689, 0.5179362
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5859828, 0.5891356
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5625920, 0.5609560
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6650190, 0.6697493
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4669266, 0.4698215
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3834605, 0.3804584
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6420259, 0.6411977
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3701732, 0.3693376

Time for backsubstitution: 20.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 567

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 567

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2118091, upper bound: 0.2112390
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2122677, upper bound: 0.2107801
time: 3.25 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 27.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.30
Output dim: 9, lower bound: -0.2107797, upper bound: 0.2122675
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.30
Output dim: 9, lower bound: -0.2112385, upper bound: 0.2118096
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 27.30
Output dim: 9, lower bound: -0.2118091, upper bound: 0.2112390
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 27.30
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

Time for backsubstitution: 21.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 2487

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1095

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2107440, upper bound: 0.2118544
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2103653, upper bound: 0.2122325
time: 3.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2

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

Time for backsubstitution: 20.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 1734

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 669

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2074295, upper bound: 0.2100586
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2094880, upper bound: 0.2080006
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

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

Time for backsubstitution: 21.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2073788, upper bound: 0.2093267
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2098928, upper bound: 0.2068115
time: 3.13 seconds

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

Time for backsubstitution: 21.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1443

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 954

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2070183, upper bound: 0.2081596
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2096487, upper bound: 0.2055302
time: 6.15 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.26 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2107440, upper bound: 0.2118544
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2103653, upper bound: 0.2122325
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2074295, upper bound: 0.2100586
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2094880, upper bound: 0.2080006
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2073788, upper bound: 0.2093267
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2098928, upper bound: 0.2068115
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2070183, upper bound: 0.2081596
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.26
Output dim: 9, lower bound: -0.2096487, upper bound: 0.2055302

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7572823, 0.7573757
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4764147, 0.4801741
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5177989, 0.5180635
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5862317, 0.5825052
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5597496, 0.5625939
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6671743, 0.6613317
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4687972, 0.4665546
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3796265, 0.3825576
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6397080, 0.6393347
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3687994, 0.3699832

Time for backsubstitution: 21.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 610

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2063193, upper bound: 0.2099611
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2088356, upper bound: 0.2074324
time: 3.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7574949, 0.7571635
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4788265, 0.4777622
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5177774, 0.5180845
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5857944, 0.5829425
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5607681, 0.5615749
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6666937, 0.6618114
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4688182, 0.4665337
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3795922, 0.3825920
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6387544, 0.6402884
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3688045, 0.3699780

Time for backsubstitution: 21.28 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1443

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2096134, upper bound: 0.2121726
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2103037, upper bound: 0.2114907
time: 3.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7179737, 0.7218080
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4875107, 0.4905539
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5054731, 0.5065522
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5913749, 0.5852213
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5132999, 0.5251660
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6439958, 0.6384530
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4510508, 0.4523792
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3779221, 0.3813608
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6010370, 0.5944781
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3589213, 0.3600774

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 610

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2073306, upper bound: 0.2099635
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2073369, upper bound: 0.2099572
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7250643, 0.7147174
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4880624, 0.4900026
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5063725, 0.5056529
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5882378, 0.5883584
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5237193, 0.5147467
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6425514, 0.6398973
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4559054, 0.4475245
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3783214, 0.3809614
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.5934014, 0.6021137
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3595850, 0.3594131

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 954

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2081

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2083120, upper bound: 0.2055166
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2070084, upper bound: 0.2068183
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1443

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 421

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2059407, upper bound: 0.2092072
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2072631, upper bound: 0.2078873
time: 4.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 22.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 1443

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 669

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2060430, upper bound: 0.2047912
time: 3.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2081457, upper bound: 0.2036262
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7571125, 0.7578154
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4680381, 0.4656100
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5088801, 0.5044580
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5877047, 0.5813098
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5577059, 0.5576434
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6605029, 0.6665840
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4610987, 0.4686058
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3734527, 0.3610492
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6372132, 0.6352029
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3640490, 0.3652558

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2909

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 669

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2032095, upper bound: 0.2064093
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2052679, upper bound: 0.2043507
time: 3.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7576962, 0.7572322
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4669576, 0.4666905
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5047436, 0.5085940
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5780201, 0.5909944
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5594683, 0.5558805
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6612220, 0.6658645
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4663420, 0.4633620
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3640149, 0.3704870
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6357832, 0.6366334
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3664343, 0.3628702

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 2081

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1095

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2096192, upper bound: 0.2050654
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2094687, upper bound: 0.2054993
time: 3.34 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.76 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2063193, upper bound: 0.2099611
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2088356, upper bound: 0.2074324
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2096134, upper bound: 0.2121726
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2103037, upper bound: 0.2114907
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2073306, upper bound: 0.2099635
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2073369, upper bound: 0.2099572
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2083120, upper bound: 0.2055166
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2070084, upper bound: 0.2068183
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2059407, upper bound: 0.2092072
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2072631, upper bound: 0.2078873
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2060430, upper bound: 0.2047912
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2081457, upper bound: 0.2036262
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2032095, upper bound: 0.2064093
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2052679, upper bound: 0.2043507
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2096192, upper bound: 0.2050654
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.76
Output dim: 9, lower bound: -0.2094687, upper bound: 0.2054993

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 564

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1443

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2055836, upper bound: 0.2099021
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2062602, upper bound: 0.2092396
time: 3.44 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 21.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1443
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 564

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2081

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2075207, upper bound: 0.2049779
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2062712, upper bound: 0.2062380
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7562437, 0.7557673
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4870610, 0.4883204
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5173645, 0.5175714
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5889435, 0.5854526
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5613227, 0.5630870
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6693387, 0.6641307
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4680886, 0.4658089
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3806765, 0.3832839
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6418982, 0.6423316
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3687229, 0.3699520

Time for backsubstitution: 21.69 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 669
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 417

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2487

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2085493, upper bound: 0.2115656
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2090413, upper bound: 0.2110752
time: 4.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7558870, 0.7561240
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4869723, 0.4884090
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5172853, 0.5176501
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5887423, 0.5856538
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5612617, 0.5631480
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6694922, 0.6639776
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4680719, 0.4658256
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3803182, 0.3836420
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6417513, 0.6424785
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3687735, 0.3699017

Time for backsubstitution: 21.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 954
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2487
type: DSZ, layer: 3, pos: 564
type: DSZ, layer: 3, pos: 1243
type: DSZ, layer: 3, pos: 1095
type: DSZ, layer: 3, pos: 417
type: DSZ, layer: 3, pos: 1734
type: DSZ, layer: 3, pos: 711
type: DSZ, layer: 3, pos: 610
type: DSZ, layer: 3, pos: 2081
type: DSZ, layer: 3, pos: 1978
type: DSZ, layer: 3, pos: 421
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 669

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 954

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.2050092, upper bound: 0.2088747
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.2079252, upper bound: 0.2062751
time: 3.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.6758566, -12.3855419, -13.6758566, -12.3855419, -0.7598333, 0.7565770
1: -14.4163694, -13.4683704, -14.4163694, -13.4683704, -0.4866734, 0.4891629
2: -8.8918123, -8.1649113, -8.8918123, -8.1649113, -0.5179591, 0.5181308
3: -8.3381863, -7.4073029, -8.3381863, -7.4073029, -0.5888395, 0.5858188
4: -1.9157456, -1.1083348, -1.9157456, -1.1083348, -0.5609078, 0.5623317
5: -11.0210896, -9.9599857, -11.0210896, -9.9599857, -0.6685944, 0.6644969
6: -13.3616886, -12.4410114, -13.3616886, -12.4410114, -0.4694557, 0.4659171
7: -3.2338426, -2.1990676, -3.2338426, -2.1990676, -0.3804018, 0.3834403
8: -5.0650358, -4.1377459, -5.0650358, -4.1377459, -0.6408138, 0.6418934
9: 4.2980032, 5.0196934, 4.2980032, 5.0196934, -0.3690798, 0.3695724

Time for backsubstitution: 21.80 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.15 + 547.31 = 606.46 seconds
