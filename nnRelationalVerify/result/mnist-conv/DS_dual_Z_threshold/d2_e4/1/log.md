## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5346226540000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0611053, 1.0611053)
1: (-9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1834817, 1.1834817)
2: (-4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4875827, 1.4875827)
3: (-11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4724336, 1.4724336)
4: (-8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5915928, 1.5915928)
5: (-0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3831000, 1.3831000)
6: (5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2248650, 1.2248650)
7: (-18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1333981, 1.1333976)
8: (-1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7780180, 1.7780180)
9: (-8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0668039, 1.0668039)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.40 + 33.88 = 57.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.5373092, upper bound: 0.5373085

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6221
type: DSZ, layer: 1, pos: 6196
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6221

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373064, upper bound: 0.5359717
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5359724, upper bound: 0.5373057
time: 3.92 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.93 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.93
Output dim: 6, lower bound: -0.5373064, upper bound: 0.5359717
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.93
Output dim: 6, lower bound: -0.5359724, upper bound: 0.5373057

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0609832, 1.0610657
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1819215, 1.1829720
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4858313, 1.4870138
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4711366, 1.4720101
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5904827, 1.5912299
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3803649, 1.3822083
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2248106, 1.2248471
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1266317, 1.1311889
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7774296, 1.7778258
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0659862, 1.0665364

Time for backsubstitution: 21.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6196
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6196

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373058, upper bound: 0.5355961
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5369308, upper bound: 0.5359711
time: 3.63 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0610657, 1.0609832
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1829724, 1.1819220
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4870133, 1.4858313
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4720101, 1.4711366
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5912299, 1.5904827
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3822083, 1.3803649
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2248468, 1.2248104
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1311884, 1.1266313
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7778254, 1.7774301
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0665364, 1.0659862

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6196
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6196

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5359718, upper bound: 0.5369301
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5355969, upper bound: 0.5373051
time: 3.82 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 30.08 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.08
Output dim: 6, lower bound: -0.5373058, upper bound: 0.5355961
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.08
Output dim: 6, lower bound: -0.5369308, upper bound: 0.5359711
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 30.08
Output dim: 6, lower bound: -0.5359718, upper bound: 0.5369301
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 30.08
Output dim: 6, lower bound: -0.5355969, upper bound: 0.5373051

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0530601, 1.0520113
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1563876, 1.1606283
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4682260, 1.4668922
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4661431, 1.4676409
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5871849, 1.5857439
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3789182, 1.3816118
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2276249, 1.2282362
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1224732, 1.1255398
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7581758, 1.7614045
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0388699, 1.0346651

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328640, upper bound: 0.5355927
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373024, upper bound: 0.5311543
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0519290, 1.0531418
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1595776, 1.1574385
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4657097, 1.4694085
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4667673, 1.4670167
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5849967, 1.5879321
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3797684, 1.3807621
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2281995, 1.2276611
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1209826, 1.1270299
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7610083, 1.7585721
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0341148, 1.0394197

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5324890, upper bound: 0.5359679
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5369275, upper bound: 0.5315293
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0531421, 1.0519292
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1574385, 1.1595778
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4694085, 1.4657097
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4670167, 1.4667673
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5879321, 1.5849967
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3807621, 1.3797684
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2276616, 1.2281995
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1270299, 1.1209826
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7585726, 1.7610087
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0394197, 1.0341148

Time for backsubstitution: 22.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5315300, upper bound: 0.5369267
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5359685, upper bound: 0.5324883
time: 3.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0520110, 1.0530598
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1606286, 1.1563880
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4668922, 1.4682260
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4676409, 1.4661431
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5857439, 1.5871849
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3816118, 1.3789182
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2282362, 1.2276249
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1255398, 1.1224732
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7614050, 1.7581763
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0346651, 1.0388699

Time for backsubstitution: 22.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5311550, upper bound: 0.5373022
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5355935, upper bound: 0.5328633
time: 3.53 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.71 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5328640, upper bound: 0.5355927
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5373024, upper bound: 0.5311543
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5324890, upper bound: 0.5359679
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5369275, upper bound: 0.5315293
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5315300, upper bound: 0.5369267
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5359685, upper bound: 0.5324883
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5311550, upper bound: 0.5373022
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.71
Output dim: 6, lower bound: -0.5355935, upper bound: 0.5328633

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0517974, 1.0470808
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1562221, 1.1599774
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4637179, 1.4657326
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4659467, 1.4668827
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5852656, 1.5782328
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3788528, 1.3813653
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2234817, 1.2271736
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1194148, 1.1247520
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7570915, 1.7571774
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0369735, 1.0272980

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328609, upper bound: 0.5347509
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5320221, upper bound: 0.5355898
time: 3.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0481286, 1.0507495
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1557376, 1.1604624
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4670668, 1.4623842
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4653854, 1.4674439
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5796733, 1.5838251
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3786716, 1.3815465
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2265620, 1.2240927
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1216855, 1.1224813
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7539492, 1.7603202
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0315022, 1.0327692

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5372993, upper bound: 0.5303125
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5364605, upper bound: 0.5311514
time: 3.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0506673, 1.0482113
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1594121, 1.1567876
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4612017, 1.4682493
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4665704, 1.4662590
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5830774, 1.5804210
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3797030, 1.3805151
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2240562, 1.2265990
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1179242, 1.1262426
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7599239, 1.7543449
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0322189, 1.0320525

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5324859, upper bound: 0.5351260
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5316471, upper bound: 0.5359649
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0469985, 1.0518796
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1589267, 1.1572726
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4645500, 1.4649005
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4660091, 1.4668202
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5774851, 1.5860133
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3795218, 1.3806963
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2271371, 1.2235181
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1201949, 1.1239719
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7567816, 1.7574878
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0267477, 1.0375237

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5369243, upper bound: 0.5306875
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5360855, upper bound: 0.5315264
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0518794, 1.0469987
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1572731, 1.1589270
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4649005, 1.4645505
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4668202, 1.4660091
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5860133, 1.5774856
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3806963, 1.3795218
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2235179, 1.2271369
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1239719, 1.1201949
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7574883, 1.7567816
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0375237, 1.0267477

Time for backsubstitution: 22.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5315269, upper bound: 0.5360851
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5306882, upper bound: 0.5369237
time: 3.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0482116, 1.0506670
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1567876, 1.1594119
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4682493, 1.4612017
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4662590, 1.4665704
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5804210, 1.5830779
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3805151, 1.3797030
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2265987, 1.2240560
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1262426, 1.1179242
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7543449, 1.7599244
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0320525, 1.0322189

Time for backsubstitution: 22.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5359654, upper bound: 0.5316464
time: 3.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5351266, upper bound: 0.5324855
time: 3.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0507493, 1.0481288
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1604621, 1.1557372
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4623842, 1.4670668
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4674439, 1.4653854
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5838251, 1.5796733
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3815465, 1.3786716
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2240930, 1.2265623
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1224813, 1.1216855
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7603207, 1.7539492
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0327692, 1.0315022

Time for backsubstitution: 22.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5311519, upper bound: 0.5364598
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5303132, upper bound: 0.5372988
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0470805, 1.0517976
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1599777, 1.1562221
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4657326, 1.4637179
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4668827, 1.4659467
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5782328, 1.5852656
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3813653, 1.3788528
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2271733, 1.2234814
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1247520, 1.1194148
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7571774, 1.7570920
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0272980, 1.0369735

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5355904, upper bound: 0.5320215
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5347516, upper bound: 0.5328603
time: 3.90 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5328609, upper bound: 0.5347509
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5320221, upper bound: 0.5355898
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5372993, upper bound: 0.5303125
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5364605, upper bound: 0.5311514
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5324859, upper bound: 0.5351260
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5316471, upper bound: 0.5359649
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5369243, upper bound: 0.5306875
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5360855, upper bound: 0.5315264
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5315269, upper bound: 0.5360851
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5306882, upper bound: 0.5369237
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5359654, upper bound: 0.5316464
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5351266, upper bound: 0.5324855
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5311519, upper bound: 0.5364598
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5303132, upper bound: 0.5372988
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5355904, upper bound: 0.5320215
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 6, lower bound: -0.5347516, upper bound: 0.5328603

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0496707, 1.0446496
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1566141, 1.1604300
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4592595, 1.4618301
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4675245, 1.4682136
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5757980, 1.5704155
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3778853, 1.3801279
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2219710, 1.2254488
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1146770, 1.1208029
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7595668, 1.7592392
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0378528, 1.0283566

Time for backsubstitution: 23.20 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1482
type: DSZ, layer: 3, pos: 177
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1831
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1738
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 1740
type: DSZ, layer: 3, pos: 3126
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 620
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2535
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 3099
type: DSZ, layer: 3, pos: 1842
type: DSZ, layer: 3, pos: 1476
type: DSZ, layer: 3, pos: 428
type: DSZ, layer: 3, pos: 202

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 3, pos: 1482

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5247233, upper bound: 0.5304712
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5285864, upper bound: 0.5266116
time: 4.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0493665, 1.0449538
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1566746, 1.1603694
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4598155, 1.4612741
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4672775, 1.4684610
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5774488, 1.5687647
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3776159, 1.3803973
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2217569, 1.2256634
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1154652, 1.1200142
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7591538, 1.7596526
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0380321, 1.0281773

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1482
type: DSZ, layer: 3, pos: 177
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1831
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1738
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 1740
type: DSZ, layer: 3, pos: 3126
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 620
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2535
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 3099
type: DSZ, layer: 3, pos: 1842
type: DSZ, layer: 3, pos: 1476
type: DSZ, layer: 3, pos: 428
type: DSZ, layer: 3, pos: 202

Time for candidate selection: 0.62 seconds

### Candidate
type: DSZ, layer: 3, pos: 1482

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5238860, upper bound: 0.5313164
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5277425, upper bound: 0.5274491
time: 4.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0460019, 1.0483179
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1561291, 1.1609149
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4626083, 1.4584818
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4669633, 1.4687748
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5702057, 1.5760078
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3777041, 1.3803091
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2250524, 1.2223680
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1169477, 1.1185322
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7564244, 1.7623820
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0323815, 1.0338278

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1482
type: DSZ, layer: 3, pos: 177
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1831
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1738
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 1740
type: DSZ, layer: 3, pos: 3126
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 620
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2535
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 3099
type: DSZ, layer: 3, pos: 1842
type: DSZ, layer: 3, pos: 1476
type: DSZ, layer: 3, pos: 428
type: DSZ, layer: 3, pos: 202

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 1482

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5291549, upper bound: 0.5260327
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5330250, upper bound: 0.5221772
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0456977, 1.0486226
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1561897, 1.1608543
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4631643, 1.4579258
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4667163, 1.4690223
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5718565, 1.5743570
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3774347, 1.3805785
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2248378, 1.2225826
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1177359, 1.1177435
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7560115, 1.7627950
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0325613, 1.0336480

Time for backsubstitution: 22.16 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1482
type: DSZ, layer: 3, pos: 177
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1831
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1738
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 1740
type: DSZ, layer: 3, pos: 3126
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 620
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2535
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 3099
type: DSZ, layer: 3, pos: 1842
type: DSZ, layer: 3, pos: 1476
type: DSZ, layer: 3, pos: 428
type: DSZ, layer: 3, pos: 202

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 1482

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5283176, upper bound: 0.5268766
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5321811, upper bound: 0.5230159
time: 3.71 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 29.82 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5247233, upper bound: 0.5304712
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5285864, upper bound: 0.5266116
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5238860, upper bound: 0.5313164
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5277425, upper bound: 0.5274491
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5291549, upper bound: 0.5260327
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5330250, upper bound: 0.5221772
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5283176, upper bound: 0.5268766
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 29.82
Output dim: 6, lower bound: -0.5321811, upper bound: 0.5230159
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5324859, upper bound: 0.5351260
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5316471, upper bound: 0.5359649
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5369243, upper bound: 0.5306875
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5360855, upper bound: 0.5315264
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5315269, upper bound: 0.5360851
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5306882, upper bound: 0.5369237
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5359654, upper bound: 0.5316464
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5351266, upper bound: 0.5324855
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5311519, upper bound: 0.5364598
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5303132, upper bound: 0.5372988
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5355904, upper bound: 0.5320215
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 29.82
Output dim: 6, lower bound: -0.5347516, upper bound: 0.5328603

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.28 + 549.72 = 607.00 seconds
