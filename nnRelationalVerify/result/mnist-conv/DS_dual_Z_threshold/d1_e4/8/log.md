## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.144541844


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3713887, 0.3713887)
1: (-4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4484067, 0.4484067)
2: (0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4432204, 0.4432206)
3: (-3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3687387, 0.3687387)
4: (-3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3927388, 0.3927386)
5: (-13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4323056, 0.4323056)
6: (-12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171718, 0.6171718)
7: (1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3184681, 0.3184681)
8: (-2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539795, 0.4539795)
9: (-5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4150467, 0.4150467)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.92 + 34.84 = 57.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.1571105, upper bound: 0.1571109

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4629
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 4629

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570969, upper bound: 0.1568652
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568648, upper bound: 0.1570967
time: 4.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.03
Output dim: 2, lower bound: -0.1570969, upper bound: 0.1568652
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.03
Output dim: 2, lower bound: -0.1568648, upper bound: 0.1570967

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3716421, 0.3712888
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4482360, 0.4488535
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4436188, 0.4430661
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3685193, 0.3693047
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3937135, 0.3923628
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4327114, 0.4321499
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171875, 0.6171665
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3183968, 0.3186536
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539599, 0.4540296
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4143586, 0.4168291

Time for backsubstitution: 20.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565212, upper bound: 0.1568512
time: 5.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1562892
time: 4.95 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3712885, 0.3713887
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4484067, 0.4482360
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4430664, 0.4432206
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3687387, 0.3685193
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3923631, 0.3927386
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4321499, 0.4323056
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171665, 0.6171718
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3184681, 0.3183968
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539795, 0.4539599
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4150467, 0.4143586

Time for backsubstitution: 21.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562891, upper bound: 0.1570837
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568515, upper bound: 0.1565217
time: 3.60 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 28.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.66
Output dim: 2, lower bound: -0.1565212, upper bound: 0.1568512
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.66
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1562892
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 28.66
Output dim: 2, lower bound: -0.1562891, upper bound: 0.1570837
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 28.66
Output dim: 2, lower bound: -0.1568515, upper bound: 0.1565217

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3712435, 0.3719776
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4479618, 0.4493227
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4430623, 0.4440250
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3684368, 0.3694456
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3938770, 0.3922675
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4324648, 0.4325731
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170998, 0.6173162
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3192058, 0.3181841
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539146, 0.4541125
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4142561, 0.4170046

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 4584

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1564306, upper bound: 0.1568467
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565162, upper bound: 0.1567609
time: 8.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3716421, 0.3708901
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4482360, 0.4485793
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4436188, 0.4425101
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3685193, 0.3692222
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3936176, 0.3923628
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4327114, 0.4319034
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171875, 0.6170793
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3179274, 0.3186536
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539599, 0.4539838
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4143586, 0.4167266

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 4584

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1569930, upper bound: 0.1562841
time: 5.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570783, upper bound: 0.1561991
time: 3.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3708901, 0.3720775
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4481320, 0.4487052
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4425101, 0.4441791
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3686559, 0.3686602
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3925266, 0.3926439
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4319034, 0.4327288
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170793, 0.6173215
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3192770, 0.3179274
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539332, 0.4540429
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4149432, 0.4145341

Time for backsubstitution: 21.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 4584

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1561985, upper bound: 0.1570788
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562841, upper bound: 0.1569928
time: 14.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3712885, 0.3709900
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4484067, 0.4479618
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4430664, 0.4426641
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3687387, 0.3684366
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3922672, 0.3927386
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4321499, 0.4320590
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171665, 0.6170845
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3179989, 0.3183968
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539795, 0.4539146
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4150467, 0.4142561

Time for backsubstitution: 22.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4584

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1567609, upper bound: 0.1565167
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568462, upper bound: 0.1564311
time: 3.86 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.34 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1564306, upper bound: 0.1568467
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1565162, upper bound: 0.1567609
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1569930, upper bound: 0.1562841
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1570783, upper bound: 0.1561991
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1561985, upper bound: 0.1570788
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1562841, upper bound: 0.1569928
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1567609, upper bound: 0.1565167
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.34
Output dim: 2, lower bound: -0.1568462, upper bound: 0.1564311

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3710973, 0.3724518
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4481087, 0.4492774
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4429779, 0.4443145
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3688526, 0.3693175
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3937674, 0.3926229
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4325237, 0.4325538
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6169529, 0.6178036
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3195887, 0.3180654
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4538283, 0.4543920
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4144416, 0.4169464

Time for backsubstitution: 21.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.42 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1514129, upper bound: 0.1526110
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1521891, upper bound: 0.1518351
time: 3.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3712435, 0.3718314
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4479165, 0.4493227
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4430623, 0.4439406
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3683085, 0.3694456
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3938770, 0.3921576
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4324455, 0.4325731
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170998, 0.6171694
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3190868, 0.3181841
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539146, 0.4540262
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4141984, 0.4170046

Time for backsubstitution: 21.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1514981, upper bound: 0.1525263
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1522743, upper bound: 0.1517502
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3714960, 0.3713644
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4483833, 0.4485340
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4435353, 0.4427996
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3689351, 0.3690941
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3935080, 0.3927183
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4327700, 0.4318841
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170411, 0.6175661
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3183107, 0.3185349
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4538746, 0.4542632
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4145446, 0.4166689

Time for backsubstitution: 21.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1519822, upper bound: 0.1520420
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1527584, upper bound: 0.1512661
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3716421, 0.3707440
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4481907, 0.4485793
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4436188, 0.4424257
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3683910, 0.3692222
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3936176, 0.3922529
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4326918, 0.4319034
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171875, 0.6169319
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3178086, 0.3186536
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539599, 0.4538980
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4143004, 0.4167266

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1520671, upper bound: 0.1519570
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1528433, upper bound: 0.1511809
time: 3.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3707438, 0.3725517
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4482794, 0.4486599
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4424257, 0.4444685
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3690720, 0.3685322
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3924170, 0.3929994
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4319623, 0.4327104
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6169319, 0.6178088
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3196602, 0.3178084
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4538479, 0.4543223
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4151292, 0.4144764

Time for backsubstitution: 21.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1511806, upper bound: 0.1528433
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1519568, upper bound: 0.1520673
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3708901, 0.3719313
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4480867, 0.4487052
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4425101, 0.4440947
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3685279, 0.3686602
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3925266, 0.3925340
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4318841, 0.4327288
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170793, 0.6171746
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3191581, 0.3179274
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539332, 0.4539571
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4148850, 0.4145341

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1512659, upper bound: 0.1527585
time: 3.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1520421, upper bound: 0.1519824
time: 3.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3711424, 0.3714643
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4485536, 0.4479165
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4429829, 0.4429536
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3691545, 0.3683088
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3921576, 0.3930945
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4322085, 0.4320407
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170206, 0.6175718
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3183823, 0.3182781
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4538937, 0.4541941
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4152322, 0.4141979

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1517499, upper bound: 0.1522743
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1525261, upper bound: 0.1514984
time: 3.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3712885, 0.3708439
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4483609, 0.4479618
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4430664, 0.4425797
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3686104, 0.3684366
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3922672, 0.3926291
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4321303, 0.4320590
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171665, 0.6169372
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3178799, 0.3183968
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539795, 0.4538283
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4149885, 0.4142561

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 3, pos: 2235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1518349, upper bound: 0.1521893
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1526111, upper bound: 0.1514132
time: 3.18 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 28.48 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1514129, upper bound: 0.1526110
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1521891, upper bound: 0.1518351
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1514981, upper bound: 0.1525263
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1522743, upper bound: 0.1517502
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1519822, upper bound: 0.1520420
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1527584, upper bound: 0.1512661
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1520671, upper bound: 0.1519570
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1528433, upper bound: 0.1511809
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1511806, upper bound: 0.1528433
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1519568, upper bound: 0.1520673
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1512659, upper bound: 0.1527585
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1520421, upper bound: 0.1519824
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1517499, upper bound: 0.1522743
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1525261, upper bound: 0.1514984
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1518349, upper bound: 0.1521893
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.48
Output dim: 2, lower bound: -0.1526111, upper bound: 0.1514132

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3685753, 0.3692677
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4479780, 0.4491439
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4382710, 0.4407644
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3688438, 0.3693132
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3916612, 0.3917723
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4316494, 0.4319811
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6157885, 0.6156793
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3191483, 0.3161182
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4514256, 0.4517722
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4139481, 0.4166269

Time for backsubstitution: 21.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1469401, upper bound: 0.1503447
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1495631, upper bound: 0.1486727
time: 3.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3710973, 0.3699298
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4479756, 0.4492774
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4429779, 0.4396076
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3688526, 0.3693087
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3929162, 0.3926229
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4325237, 0.4316795
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6169529, 0.6166391
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3176417, 0.3180654
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4538283, 0.4519892
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4141221, 0.4169464

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1482505, upper bound: 0.1499855
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1499223, upper bound: 0.1473625
time: 3.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3687224, 0.3686473
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4477854, 0.4491887
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4383554, 0.4403906
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3683000, 0.3694413
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3917708, 0.3913069
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4315712, 0.4320004
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6159353, 0.6150451
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3186462, 0.3162374
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4515119, 0.4514070
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4137044, 0.4166846

Time for backsubstitution: 21.88 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1470253, upper bound: 0.1502598
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1496484, upper bound: 0.1485878
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3712435, 0.3693094
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4477830, 0.4493227
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4430623, 0.4392338
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3683085, 0.3694367
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3930264, 0.3921576
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4324455, 0.4316988
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170998, 0.6160049
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3171396, 0.3181841
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539146, 0.4516234
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4138784, 0.4170046

Time for backsubstitution: 21.89 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1483357, upper bound: 0.1499006
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1500076, upper bound: 0.1472775
time: 2.98 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 28.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1469401, upper bound: 0.1503447
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1495631, upper bound: 0.1486727
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1482505, upper bound: 0.1499855
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1499223, upper bound: 0.1473625
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1470253, upper bound: 0.1502598
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1496484, upper bound: 0.1485878
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1483357, upper bound: 0.1499006
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 28.17
Output dim: 2, lower bound: -0.1500076, upper bound: 0.1472775
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1519822, upper bound: 0.1520420
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1527584, upper bound: 0.1512661
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1520671, upper bound: 0.1519570
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1528433, upper bound: 0.1511809
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1511806, upper bound: 0.1528433
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1519568, upper bound: 0.1520673
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1512659, upper bound: 0.1527585
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1520421, upper bound: 0.1519824
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1517499, upper bound: 0.1522743
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1525261, upper bound: 0.1514984
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1518349, upper bound: 0.1521893
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 28.17
Output dim: 2, lower bound: -0.1526111, upper bound: 0.1514132

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.76 + 547.65 = 605.41 seconds
