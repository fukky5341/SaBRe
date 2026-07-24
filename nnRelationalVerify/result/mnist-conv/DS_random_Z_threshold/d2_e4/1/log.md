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
execution time: IAR + RelationalAnalysis = 24.23 + 33.07 = 57.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.5373092, upper bound: 0.5373085

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 6196
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 481

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373052
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5328667
time: 3.35 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.88
Output dim: 6, lower bound: -0.5328674, upper bound: 0.5373052
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.88
Output dim: 6, lower bound: -0.5373059, upper bound: 0.5328667

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0598431, 1.0561748
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1833158, 1.1828308
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4830751, 1.4864240
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4722371, 1.4716759
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5896740, 1.5840816
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3830342, 1.3828530
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2207212, 1.2238023
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1303391, 1.1326103
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7769337, 1.7737908
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0649085, 1.0594378

Time for backsubstitution: 22.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6196
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6196

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328669, upper bound: 0.5369297
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5324919, upper bound: 0.5373046
time: 3.67 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0561748, 1.0598431
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1828308, 1.1833158
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4864240, 1.4830751
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4716759, 1.4722371
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5840816, 1.5896740
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3828530, 1.3830342
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2238026, 1.2207215
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1326098, 1.1303396
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7737913, 1.7769337
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0594373, 1.0649090

Time for backsubstitution: 23.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 6196
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5373028, upper bound: 0.5320248
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5364639, upper bound: 0.5328637
time: 3.25 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.65 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 6, lower bound: -0.5328669, upper bound: 0.5369297
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 6, lower bound: -0.5324919, upper bound: 0.5373046
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 6, lower bound: -0.5373028, upper bound: 0.5320248
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.65
Output dim: 6, lower bound: -0.5364639, upper bound: 0.5328637

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0519199, 1.0471206
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1577823, 1.1604872
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4654703, 1.4663024
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4672427, 1.4673052
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5863762, 1.5785956
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3815885, 1.3822570
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2235360, 1.2271914
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1261816, 1.1269617
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7576790, 1.7573695
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0377908, 1.0275650

Time for backsubstitution: 22.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328638, upper bound: 0.5360877
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5320250, upper bound: 0.5369265
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0507894, 1.0482512
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1609719, 1.1572971
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4629536, 1.4688191
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4678664, 1.4666815
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5841880, 1.5807838
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3824382, 1.3814073
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2241106, 1.2266169
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1246910, 1.1284523
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7605124, 1.7545371
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0330362, 1.0323200

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6221
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5324890, upper bound: 0.5359679
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5311550, upper bound: 0.5373022
time: 3.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0540481, 1.0574119
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1832228, 1.1837683
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4819651, 1.4791722
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4732533, 1.4735675
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5746145, 1.5818577
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3818860, 1.3817978
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2222915, 1.2189963
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1278720, 1.1263900
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7762666, 1.7789960
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0603161, 1.0659666

Time for backsubstitution: 23.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6221
type: DSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5372999, upper bound: 0.5306880
time: 3.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5359659, upper bound: 0.5320222
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0537438, 1.0577161
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1832829, 1.1837072
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4825211, 1.4786162
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4730062, 1.4738145
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5762653, 1.5802069
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3816166, 1.3820672
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2220774, 1.2192109
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1286602, 1.1256013
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7758536, 1.7794094
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0604954, 1.0657873

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6196
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6196

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5364634, upper bound: 0.5324882
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5360884, upper bound: 0.5328632
time: 3.71 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5328638, upper bound: 0.5360877
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5320250, upper bound: 0.5369265
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5324890, upper bound: 0.5359679
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5311550, upper bound: 0.5373022
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5372999, upper bound: 0.5306880
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5359659, upper bound: 0.5320222
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5364634, upper bound: 0.5324882
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.07
Output dim: 6, lower bound: -0.5360884, upper bound: 0.5328632

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0497928, 1.0446892
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1581743, 1.1609395
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4610124, 1.4624004
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4688215, 1.4686365
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5769086, 1.5707784
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3806205, 1.3810201
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2220254, 1.2254667
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1214442, 1.1230125
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7601542, 1.7594314
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0386701, 1.0286231

Time for backsubstitution: 22.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5328609, upper bound: 0.5347509
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5315269, upper bound: 0.5360851
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0494885, 1.0449934
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1582344, 1.1608791
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4615684, 1.4618444
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4685740, 1.4688840
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5785589, 1.5691276
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3803511, 1.3812895
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2218113, 1.2256813
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1222324, 1.1222243
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7597413, 1.7598448
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0388494, 1.0284438

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5320221, upper bound: 0.5355898
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5306882, upper bound: 0.5369237
time: 3.48 seconds

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

Time for backsubstitution: 22.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5324859, upper bound: 0.5351260
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5316471, upper bound: 0.5359649
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4558

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4558

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5311519, upper bound: 0.5364598
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5303132, upper bound: 0.5372988
time: 3.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0539260, 1.0573723
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1816621, 1.1832581
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4802117, 1.4786019
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4719567, 1.4731445
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5735040, 1.5814948
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3791509, 1.3809066
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2222376, 1.2189786
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1211061, 1.1241808
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7756782, 1.7788029
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0594988, 1.0656996

Time for backsubstitution: 23.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5372993, upper bound: 0.5303125
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5369243, upper bound: 0.5306875
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0540080, 1.0572903
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1827130, 1.1822076
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4813943, 1.4774194
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4728303, 1.4722710
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5742517, 1.5807471
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3809943, 1.3790631
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2222738, 1.2189419
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1256633, 1.1196237
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7760730, 1.7784071
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0600486, 1.0651493

Time for backsubstitution: 23.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6196

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5359654, upper bound: 0.5316464
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5355904, upper bound: 0.5320215
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0458198, 1.0486622
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1577499, 1.1613638
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4649167, 1.4584956
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4680128, 1.4694452
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5729666, 1.5747199
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3801699, 1.3814707
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2248921, 1.2226005
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1245031, 1.1199536
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7565989, 1.7629871
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0333781, 1.0339150

Time for backsubstitution: 25.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5364605, upper bound: 0.5311514
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5351266, upper bound: 0.5324855
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0446892, 1.0497928
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1609395, 1.1581740
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4624004, 1.4610124
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4686370, 1.4688215
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5707784, 1.5769086
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3810196, 1.3806205
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2254667, 1.2220254
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1230125, 1.1214442
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7594314, 1.7601547
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0286231, 1.0386701

Time for backsubstitution: 25.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5360855, upper bound: 0.5315264
time: 3.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5347516, upper bound: 0.5328603
time: 3.79 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.57 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5328609, upper bound: 0.5347509
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5315269, upper bound: 0.5360851
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5320221, upper bound: 0.5355898
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5306882, upper bound: 0.5369237
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5324859, upper bound: 0.5351260
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5316471, upper bound: 0.5359649
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5311519, upper bound: 0.5364598
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5303132, upper bound: 0.5372988
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5372993, upper bound: 0.5303125
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5369243, upper bound: 0.5306875
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5359654, upper bound: 0.5316464
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5355904, upper bound: 0.5320215
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5364605, upper bound: 0.5311514
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5351266, upper bound: 0.5324855
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.57
Output dim: 6, lower bound: -0.5360855, upper bound: 0.5315264
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.57
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

Time for backsubstitution: 22.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 1740
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 3099
type: DSZ, layer: 3, pos: 1842
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 620
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1482
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1476
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 202
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 428
type: DSZ, layer: 3, pos: 2535
type: DSZ, layer: 3, pos: 3126
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1738
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 1831
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 177
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2816

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5234665, upper bound: 0.5260912
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5240294, upper bound: 0.5253508
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0497527, 1.0445676
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1576645, 1.1593795
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4604421, 1.4606476
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4683981, 1.4673400
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5765452, 1.5696683
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3797288, 1.3782845
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2220078, 1.2254121
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1192336, 1.1162457
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7599626, 1.7588434
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0384030, 1.0278063

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 1831
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 1842
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 1740
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2535
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 1476
type: DSZ, layer: 3, pos: 202
type: DSZ, layer: 3, pos: 428
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1482
type: DSZ, layer: 3, pos: 177
type: DSZ, layer: 3, pos: 1738
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 3126
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 3099
type: DSZ, layer: 3, pos: 620
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 2082

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5300499, upper bound: 0.5347276
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.5301853, upper bound: 0.5346323
time: 3.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 22.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 177
type: DSZ, layer: 3, pos: 1411
type: DSZ, layer: 3, pos: 326
type: DSZ, layer: 3, pos: 620
type: DSZ, layer: 3, pos: 1842
type: DSZ, layer: 3, pos: 1508
type: DSZ, layer: 3, pos: 202
type: DSZ, layer: 3, pos: 1500
type: DSZ, layer: 3, pos: 2816
type: DSZ, layer: 3, pos: 3126
type: DSZ, layer: 3, pos: 1476
type: DSZ, layer: 3, pos: 1108
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1831
type: DSZ, layer: 3, pos: 3099
type: DSZ, layer: 3, pos: 2118
type: DSZ, layer: 3, pos: 2326
type: DSZ, layer: 3, pos: 1703
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 81
type: DSZ, layer: 3, pos: 1776
type: DSZ, layer: 3, pos: 1482
type: DSZ, layer: 3, pos: 1740
type: DSZ, layer: 3, pos: 1808
type: DSZ, layer: 3, pos: 2634
type: DSZ, layer: 3, pos: 2136
type: DSZ, layer: 3, pos: 599
type: DSZ, layer: 3, pos: 1850
type: DSZ, layer: 3, pos: 1158
type: DSZ, layer: 3, pos: 1738
type: DSZ, layer: 3, pos: 2131
type: DSZ, layer: 3, pos: 697
type: DSZ, layer: 3, pos: 2535
type: DSZ, layer: 3, pos: 305
type: DSZ, layer: 3, pos: 1746
type: DSZ, layer: 3, pos: 2082
type: DSZ, layer: 3, pos: 2831
type: DSZ, layer: 3, pos: 428

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 913

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5281824, upper bound: 0.5314418
time: 3.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.5275224, upper bound: 0.5319170
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7340474, -3.2502456, -4.7340474, -3.2502456, -1.0494485, 1.0448718
1: -9.6282129, -7.8908486, -9.6282129, -7.8908486, -1.1577251, 1.1593189
2: -4.8924956, -3.2923169, -4.8924956, -3.2923169, -1.4609981, 1.4600916
3: -11.5050545, -9.6220703, -11.5050545, -9.6220703, -1.4681511, 1.4675875
4: -8.0196972, -6.0275412, -8.0196972, -6.0275412, -1.5781960, 1.5680175
5: -0.4153727, 1.0425191, -0.4153727, 1.0425191, -1.3794594, 1.3785539
6: 5.8199577, 7.1746778, 5.8199577, 7.1746778, -1.2217937, 1.2256267
7: -18.3088875, -16.2116203, -18.3088875, -16.2116203, -1.1200223, 1.1154571
8: -1.0622559, 0.7300744, -1.0622559, 0.7300744, -1.7595496, 1.7592568
9: -8.3877430, -6.9174862, -8.3877430, -6.9174862, -1.0385823, 1.0276270

Time for backsubstitution: 22.54 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.30 + 542.76 = 600.06 seconds
