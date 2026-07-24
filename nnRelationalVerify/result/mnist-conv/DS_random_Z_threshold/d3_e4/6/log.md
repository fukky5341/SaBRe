## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1823463684


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5293298, 2.5293295)
1: (-10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2577815, 2.2577815)
2: (-6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.3718305, 2.3718295)
3: (-2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8441834, 1.8441832)
4: (-6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1593237, 3.1593237)
5: (-8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4321971, 2.4321966)
6: (-19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1931105, 3.1931105)
7: (4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772)
8: (-7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3909245, 2.3909245)
9: (-7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6847959, 2.6847959)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.79 + 33.81 = 57.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -1.1847179, upper bound: 1.1847154

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 478
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 457

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 478

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825689, upper bound: 1.1847139
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847147, upper bound: 1.1825679
time: 4.99 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.46
Output dim: 7, lower bound: -1.1825689, upper bound: 1.1847139
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.46
Output dim: 7, lower bound: -1.1847147, upper bound: 1.1825679

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5279202, 2.5280938
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2073236, 2.2001090
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2774968, 2.2893014
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8233452, 1.8259487
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1183290, 3.1124687
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4197197, 2.4174356
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1746912, 3.1809759
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3416762, 2.3304830
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6874561, 2.6790318

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6209

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825677, upper bound: 1.1827268
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1805822, upper bound: 1.1847127
time: 4.83 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5280938, 2.5279198
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2001090, 2.2073240
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2893014, 2.2774968
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8259492, 1.8233445
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1124687, 3.1183290
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4174366, 2.4197187
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1809759, 3.1746912
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3304834, 2.3416760
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6790314, 2.6874561

Time for backsubstitution: 23.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837852, upper bound: 1.1825606
time: 5.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847074, upper bound: 1.1816388
time: 4.53 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 33.39 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.39
Output dim: 7, lower bound: -1.1825677, upper bound: 1.1827268
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.39
Output dim: 7, lower bound: -1.1805822, upper bound: 1.1847127
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 33.39
Output dim: 7, lower bound: -1.1837852, upper bound: 1.1825606
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 33.39
Output dim: 7, lower bound: -1.1847074, upper bound: 1.1816388

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4786606, 2.4718103
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1809492, 2.1699610
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2180395, 2.2375269
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7893839, 1.7964175
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1011667, 3.0925808
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3975282, 2.3991361
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1458197, 3.1479912
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3387170, 2.3260169
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6651058, 2.6527667

Time for backsubstitution: 23.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825619, upper bound: 1.1782414
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1780843, upper bound: 1.1827209
time: 4.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4716358, 2.4788349
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1771755, 2.1737337
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2257223, 2.2298441
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7938137, 1.7919879
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0984411, 3.0953064
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4014192, 2.3952451
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1417065, 3.1521044
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3372097, 2.3275239
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6611910, 2.6566820

Time for backsubstitution: 23.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1805764, upper bound: 1.1802293
time: 6.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1761009, upper bound: 1.1847068
time: 4.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5253434, 2.5282376
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1992092, 2.2074280
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2894301, 2.2764068
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8263254, 1.8201270
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1113548, 3.1184583
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4161682, 2.4198709
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1811066, 3.1736622
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3306689, 2.3400717
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6767521, 2.6877398

Time for backsubstitution: 23.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 457

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1837793, upper bound: 1.1780774
time: 4.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1793018, upper bound: 1.1825554
time: 4.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5280938, 2.5251698
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.2001090, 2.2064242
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2882118, 2.2774968
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8227315, 1.8233445
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1124687, 3.1172147
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4174366, 2.4184523
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1799469, 3.1746912
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3288789, 2.3416760
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6790314, 2.6851764

Time for backsubstitution: 22.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6209

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1847063, upper bound: 1.1796515
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1827207, upper bound: 1.1816378
time: 5.30 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1825619, upper bound: 1.1782414
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1780843, upper bound: 1.1827209
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1805764, upper bound: 1.1802293
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1761009, upper bound: 1.1847068
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1837793, upper bound: 1.1780774
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1793018, upper bound: 1.1825554
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1847063, upper bound: 1.1796515
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.31
Output dim: 7, lower bound: -1.1827207, upper bound: 1.1816378

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4775095, 2.4711771
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1789374, 2.1663051
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2117658, 2.2340803
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7881322, 1.7957284
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0929222, 3.0775919
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3974385, 2.3990917
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1435928, 3.1467714
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3349361, 2.3239372
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6641350, 2.6509976

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825619, upper bound: 1.1748648
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791828, upper bound: 1.1782415
time: 4.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4780273, 2.4706590
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1772933, 2.1679487
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2145929, 2.2312531
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7886944, 1.7951663
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0861778, 3.0843363
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3974843, 2.3990464
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1445999, 3.1457644
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3366375, 2.3222358
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6633368, 2.6517963

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1771548, upper bound: 1.1827137
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1780771, upper bound: 1.1817914
time: 4.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4710026, 2.4776835
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1735206, 2.1717224
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2222757, 2.2235708
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7931242, 1.7907367
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0834522, 3.0870619
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4013753, 2.3951554
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1404867, 3.1498775
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3351307, 2.3237429
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6594219, 2.6557117

Time for backsubstitution: 22.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5746
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5746

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1751692, upper bound: 1.1846995
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1760915, upper bound: 1.1837772
time: 4.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5241933, 2.5276048
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1971979, 2.2037725
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2831554, 2.2729592
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8250747, 1.8194380
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.1031094, 3.1034689
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4160786, 2.4198256
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1788797, 3.1724424
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3268876, 2.3379920
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6757832, 2.6859717

Time for backsubstitution: 23.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 6209
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 73

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825257, upper bound: 1.1780746
time: 7.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1793407, upper bound: 1.1780772
time: 5.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.5247102, 2.5270867
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1955538, 2.2054162
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2859826, 2.2701321
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.8256364, 1.8188760
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0963650, 3.1102133
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4161243, 2.4197803
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1798868, 3.1714354
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3285890, 2.3362906
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6749840, 2.6867700

Time for backsubstitution: 23.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 6209

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1793017, upper bound: 1.1781162
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1792982, upper bound: 1.1813060
time: 5.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4788342, 2.4688864
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1737337, 2.1762762
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2287550, 2.2257223
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7887702, 1.7938132
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0953064, 3.0973263
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3952451, 2.4001527
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1510763, 3.1417055
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3259206, 2.3372099
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6566820, 2.6589112

Time for backsubstitution: 23.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 539

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 73

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1811581, upper bound: 1.1782173
time: 6.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1834219, upper bound: 1.1782141
time: 5.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4718103, 2.4759109
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1699610, 2.1800494
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2364378, 2.2180395
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7932000, 1.7893836
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0925808, 3.1000519
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3991361, 2.3962622
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1469622, 3.1458197
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3244133, 2.3387170
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6527672, 2.6628270

Time for backsubstitution: 21.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 457
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 73

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791609, upper bound: 1.1802027
time: 6.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1814172, upper bound: 1.1801997
time: 4.52 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 32.37 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1825619, upper bound: 1.1748648
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1791828, upper bound: 1.1782415
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1771548, upper bound: 1.1827137
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1780771, upper bound: 1.1817914
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1751692, upper bound: 1.1846995
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1760915, upper bound: 1.1837772
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1825257, upper bound: 1.1780746
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1793407, upper bound: 1.1780772
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1793017, upper bound: 1.1781162
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1792982, upper bound: 1.1813060
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1811581, upper bound: 1.1782173
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1834219, upper bound: 1.1782141
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1791609, upper bound: 1.1802027
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 32.37
Output dim: 7, lower bound: -1.1814172, upper bound: 1.1801997

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4775023, 2.4711709
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1789312, 2.1663003
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2117634, 2.2340775
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7881150, 1.7957127
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0929213, 3.0775919
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3974209, 2.3990760
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1435833, 3.1467624
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3349257, 2.3239250
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6641312, 2.6509943

Time for backsubstitution: 21.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 5746

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1825597, upper bound: 1.1748434
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1791797, upper bound: 1.1748629
time: 4.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4752769, 2.4709766
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1763940, 2.1680527
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2147222, 2.2301641
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7890711, 1.7919486
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0850620, 3.0844641
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.3962169, 2.3991971
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1447296, 3.1447353
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3368235, 2.3206320
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6610584, 2.6520801

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1771548, upper bound: 1.1793415
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1737756, upper bound: 1.1827143
time: 4.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4682522, 2.4780011
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1726203, 2.1718259
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2224045, 2.2224813
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7935009, 1.7875190
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0823364, 3.0871897
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4001079, 2.3953061
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1406164, 3.1488495
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3353162, 2.3221390
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6571426, 2.6559949

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 539
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 539

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -1.1751672, upper bound: 1.1813174
time: 7.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.1717804, upper bound: 1.1846973
time: 5.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -17.5972614, -13.5857916, -17.5972614, -13.5857916, -2.4710026, 2.4749331
1: -10.2654305, -7.4666815, -10.2654305, -7.4666815, -2.1735206, 2.1708226
2: -6.4559197, -3.5972714, -6.4559197, -3.5972714, -2.2211862, 2.2235708
3: -2.4377689, 0.1256924, -2.4377689, 0.1256924, -1.7899065, 1.7907367
4: -6.9938755, -2.8966293, -6.9938755, -2.8966293, -3.0834522, 3.0859461
5: -8.9602108, -5.7368841, -8.9602108, -5.7368841, -2.4013753, 2.3938880
6: -19.4462585, -15.5525522, -19.4462585, -15.5525522, -3.1394577, 3.1498775
7: 4.2598286, 6.9828057, 4.2598286, 6.9828057, -2.7229772, 2.7229772
8: -7.1687799, -4.4007730, -7.1687799, -4.4007730, -2.3335261, 2.3237429
9: -7.2100573, -3.7771635, -7.2100573, -3.7771635, -2.6594219, 2.6534319

Time for backsubstitution: 21.46 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.60 + 557.65 = 615.25 seconds
