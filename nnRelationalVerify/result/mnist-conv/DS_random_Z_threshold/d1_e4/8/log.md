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
execution time: IAR + RelationalAnalysis = 22.69 + 34.12 = 56.81 seconds
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

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4629

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570969, upper bound: 0.1568652
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568648, upper bound: 0.1570967
time: 4.30 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.49
Output dim: 2, lower bound: -0.1570969, upper bound: 0.1568652
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.49
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

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565212, upper bound: 0.1568512
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1562892
time: 4.81 seconds

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

Time for backsubstitution: 21.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4584
type: DSZ, layer: 1, pos: 4582

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4584

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1567742, upper bound: 0.1570915
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568598, upper bound: 0.1570068
time: 3.94 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 2, lower bound: -0.1565212, upper bound: 0.1568512
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 2, lower bound: -0.1570836, upper bound: 0.1562892
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 2, lower bound: -0.1567742, upper bound: 0.1570915
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 2, lower bound: -0.1568598, upper bound: 0.1570068

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

Time for backsubstitution: 21.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4584

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1564306, upper bound: 0.1568467
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565162, upper bound: 0.1567609
time: 8.33 seconds

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

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4584

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4584

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1569930, upper bound: 0.1562841
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1570783, upper bound: 0.1561991
time: 3.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3711424, 0.3718629
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4485536, 0.4481907
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4429829, 0.4435110
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3691545, 0.3683910
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3922534, 0.3930945
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4322085, 0.4322872
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6170206, 0.6176600
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3188517, 0.3182781
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4538937, 0.4542398
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4152322, 0.4143009

Time for backsubstitution: 21.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1561985, upper bound: 0.1570788
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1567609, upper bound: 0.1565167
time: 3.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3712885, 0.3712423
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4483609, 0.4482360
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4430664, 0.4431367
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3686104, 0.3685193
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3923631, 0.3926291
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4321303, 0.4323056
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6171665, 0.6170259
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3183496, 0.3183968
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539795, 0.4538746
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4149885, 0.4143586

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4582

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4582

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1562841, upper bound: 0.1569928
time: 13.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1568462, upper bound: 0.1564311
time: 3.39 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 39.01 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.01
Output dim: 2, lower bound: -0.1564306, upper bound: 0.1568467
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.01
Output dim: 2, lower bound: -0.1565162, upper bound: 0.1567609
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.01
Output dim: 2, lower bound: -0.1569930, upper bound: 0.1562841
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.01
Output dim: 2, lower bound: -0.1570783, upper bound: 0.1561991
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.01
Output dim: 2, lower bound: -0.1561985, upper bound: 0.1570788
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.01
Output dim: 2, lower bound: -0.1567609, upper bound: 0.1565167
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 39.01
Output dim: 2, lower bound: -0.1562841, upper bound: 0.1569928
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 39.01
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

Time for backsubstitution: 22.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 652

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1563529, upper bound: 0.1531890
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1533803, upper bound: 0.1567724
time: 4.05 seconds

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

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1410

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1564267, upper bound: 0.1553690
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1551244, upper bound: 0.1566713
time: 3.42 seconds

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

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3125

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565526, upper bound: 0.1532238
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1539321, upper bound: 0.1558436
time: 5.03 seconds

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

Time for backsubstitution: 23.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2812

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1537430, upper bound: 0.1551255
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1560052, upper bound: 0.1528630
time: 4.75 seconds

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

Time for backsubstitution: 23.93 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2370

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1797

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1557669, upper bound: 0.1566352
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1557506, upper bound: 0.1566487
time: 4.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 23.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 1202

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1488

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1565675, upper bound: 0.1548696
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1551145, upper bound: 0.1563232
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 24.78 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1556061, upper bound: 0.1568363
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1561270, upper bound: 0.1563142
time: 3.82 seconds

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

Time for backsubstitution: 24.94 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 2142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2370

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1458717, upper bound: 0.1454502
time: 2.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1458717, upper bound: 0.1454502
time: 2.93 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1563529, upper bound: 0.1531890
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1533803, upper bound: 0.1567724
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1564267, upper bound: 0.1553690
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1551244, upper bound: 0.1566713
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1565526, upper bound: 0.1532238
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1539321, upper bound: 0.1558436
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1537430, upper bound: 0.1551255
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1560052, upper bound: 0.1528630
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1557669, upper bound: 0.1566352
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1557506, upper bound: 0.1566487
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1565675, upper bound: 0.1548696
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1551145, upper bound: 0.1563232
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1556061, upper bound: 0.1568363
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1561270, upper bound: 0.1563142
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1458717, upper bound: 0.1454502
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.83
Output dim: 2, lower bound: -0.1458717, upper bound: 0.1454502

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3634727, 0.3623130
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4483333, 0.4486785
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4419427, 0.4430804
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3704832, 0.3712170
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3936038, 0.3924139
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4326613, 0.4327135
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6135812, 0.6133451
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3193192, 0.3184268
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4557953, 0.4558463
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4172058, 0.4191022

Time for backsubstitution: 24.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 1202

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1521844, upper bound: 0.1492483
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1524039, upper bound: 0.1490212
time: 3.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3618586, 0.3648272
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4478049, 0.4495015
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4418151, 0.4432788
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3707519, 0.3710985
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3935580, 0.3925169
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4326837, 0.4327035
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6129675, 0.6144319
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3199503, 0.3180218
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4554963, 0.4563589
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4168143, 0.4197106

Time for backsubstitution: 26.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 233
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 1410
type: DSZ, layer: 3, pos: 1266

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1492101, upper bound: 0.1528340
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1494295, upper bound: 0.1526069
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -13.1750603, -12.2906647, -13.1750603, -12.2906647, -0.3651257, 0.3652451
1: -4.0107384, -3.4168868, -4.0107384, -3.4168868, -0.4472356, 0.4486866
2: 0.1172709, 0.7854009, 0.1172709, 0.7854009, -0.4431629, 0.4439073
3: -3.5588059, -2.8935289, -3.5588059, -2.8935289, -0.3682470, 0.3693838
4: -3.6335125, -2.9203374, -3.6335125, -2.9203374, -0.3937168, 0.3920524
5: -13.0396671, -12.2592764, -13.0396671, -12.2592764, -0.4331367, 0.4330640
6: -12.9865103, -12.1916008, -12.9865103, -12.1916008, -0.6194496, 0.6206388
7: 1.6868830, 2.3739386, 1.6868830, 2.3739386, -0.3187776, 0.3176303
8: -2.6495357, -2.0242376, -2.6495357, -2.0242376, -0.4539194, 0.4540324
9: -5.0104799, -4.2063570, -5.0104799, -4.2063570, -0.4143081, 0.4171953

Time for backsubstitution: 24.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2142
type: DSZ, layer: 3, pos: 2235
type: DSZ, layer: 3, pos: 1488
type: DSZ, layer: 3, pos: 1250
type: DSZ, layer: 3, pos: 974
type: DSZ, layer: 3, pos: 3125
type: DSZ, layer: 3, pos: 2133
type: DSZ, layer: 3, pos: 652
type: DSZ, layer: 3, pos: 2812
type: DSZ, layer: 3, pos: 2370
type: DSZ, layer: 3, pos: 1266
type: DSZ, layer: 3, pos: 1202
type: DSZ, layer: 3, pos: 82
type: DSZ, layer: 3, pos: 1797
type: DSZ, layer: 3, pos: 2223
type: DSZ, layer: 3, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1692

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1543781, upper bound: 0.1493857
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.1505424, upper bound: 0.1533778
time: 3.57 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 32.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.56
Output dim: 2, lower bound: -0.1521844, upper bound: 0.1492483
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.56
Output dim: 2, lower bound: -0.1524039, upper bound: 0.1490212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.56
Output dim: 2, lower bound: -0.1492101, upper bound: 0.1528340
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.56
Output dim: 2, lower bound: -0.1494295, upper bound: 0.1526069
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 32.56
Output dim: 2, lower bound: -0.1543781, upper bound: 0.1493857
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 32.56
Output dim: 2, lower bound: -0.1505424, upper bound: 0.1533778
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1551244, upper bound: 0.1566713
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1565526, upper bound: 0.1532238
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1539321, upper bound: 0.1558436
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1537430, upper bound: 0.1551255
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1560052, upper bound: 0.1528630
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1557669, upper bound: 0.1566352
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1557506, upper bound: 0.1566487
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1565675, upper bound: 0.1548696
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1551145, upper bound: 0.1563232
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1556061, upper bound: 0.1568363
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1561270, upper bound: 0.1563142
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1458717, upper bound: 0.1454502
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.56
Output dim: 2, lower bound: -0.1458717, upper bound: 0.1454502

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.81 + 548.31 = 605.12 seconds
