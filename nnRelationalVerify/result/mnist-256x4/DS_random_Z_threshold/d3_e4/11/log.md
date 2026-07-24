## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 7.6259627037


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305)
1: (-4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668)
2: (-6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820)
3: (-5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251)
4: (-6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038)
5: (-4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198)
6: (-4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974)
7: (-5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517)
8: (-6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594)
9: (-4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.86 + 4.74 = 5.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6335958, upper bound: 7.6335962

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335958, upper bound: 7.6335961
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335957, upper bound: 7.6335959
time: 2.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.09
Output dim: 2, lower bound: -7.6335958, upper bound: 7.6335961
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.09
Output dim: 2, lower bound: -7.6335957, upper bound: 7.6335959

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6328362, upper bound: 7.6328366
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6328362, upper bound: 7.6328366
time: 8.90 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335960, upper bound: 7.6335959
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335961, upper bound: 7.6335954
time: 2.16 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.77 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 2, lower bound: -7.6328362, upper bound: 7.6328366
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 2, lower bound: -7.6328362, upper bound: 7.6328366
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 2, lower bound: -7.6335960, upper bound: 7.6335959
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.77
Output dim: 2, lower bound: -7.6335961, upper bound: 7.6335954

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290253
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290254
time: 2.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6328359, upper bound: 7.6328366
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6328361, upper bound: 7.6328364
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335955, upper bound: 7.6335960
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335956, upper bound: 7.6335957
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6331316, upper bound: 7.6331318
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6331316, upper bound: 7.6331318
time: 2.23 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.92 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290253
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290254
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6328359, upper bound: 7.6328366
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6328361, upper bound: 7.6328364
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6335955, upper bound: 7.6335960
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6335956, upper bound: 7.6335957
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6331316, upper bound: 7.6331318
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.92
Output dim: 2, lower bound: -7.6331316, upper bound: 7.6331318

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
time: 3.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290254
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290254
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323213, upper bound: 7.6323222
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323213, upper bound: 7.6323221
time: 4.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324320, upper bound: 7.6324318
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324320, upper bound: 7.6324318
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314493, upper bound: 7.6314501
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314493, upper bound: 7.6314501
time: 2.28 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335954, upper bound: 7.6335957
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335955, upper bound: 7.6335959
time: 4.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 65

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304891, upper bound: 7.6304891
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304891, upper bound: 7.6304891
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310127, upper bound: 7.6310127
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310127, upper bound: 7.6310127
time: 2.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290254
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6290257, upper bound: 7.6290254
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6323213, upper bound: 7.6323222
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6323213, upper bound: 7.6323221
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6324320, upper bound: 7.6324318
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6324320, upper bound: 7.6324318
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6314493, upper bound: 7.6314501
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6314493, upper bound: 7.6314501
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6335954, upper bound: 7.6335957
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6335955, upper bound: 7.6335959
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6304891, upper bound: 7.6304891
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6304891, upper bound: 7.6304891
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310127, upper bound: 7.6310127
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310127, upper bound: 7.6310127

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289833, upper bound: 7.6289833
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289831
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289832
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290258, upper bound: 7.6290252
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290258, upper bound: 7.6290253
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6122348, upper bound: 7.6122369
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6122348, upper bound: 7.6122361
time: 2.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316841, upper bound: 7.6316843
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316841, upper bound: 7.6316843
time: 2.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324313, upper bound: 7.6324313
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324313
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324162, upper bound: 7.6324166
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324162, upper bound: 7.6324160
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314394, upper bound: 7.6314402
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314394, upper bound: 7.6314401
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310129, upper bound: 7.6310136
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310129, upper bound: 7.6310127
time: 3.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335811, upper bound: 7.6335815
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335811, upper bound: 7.6335812
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335577, upper bound: 7.6335582
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335577, upper bound: 7.6335584
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4194101, upper bound: 7.4194108
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4194101, upper bound: 7.4194108
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277781, upper bound: 7.6277798
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277781, upper bound: 7.6277798
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309302
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309302
time: 2.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309302
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309301
time: 3.81 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 7.05 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6289833, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289832
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6290258, upper bound: 7.6290252
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6290258, upper bound: 7.6290253
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6122348, upper bound: 7.6122369
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6122348, upper bound: 7.6122361
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6316841, upper bound: 7.6316843
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6316841, upper bound: 7.6316843
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6324313, upper bound: 7.6324313
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324313
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6324162, upper bound: 7.6324166
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6324162, upper bound: 7.6324160
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6314394, upper bound: 7.6314402
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6314394, upper bound: 7.6314401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6310129, upper bound: 7.6310136
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6310129, upper bound: 7.6310127
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6335811, upper bound: 7.6335815
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6335811, upper bound: 7.6335812
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6335577, upper bound: 7.6335582
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6335577, upper bound: 7.6335584
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.4194101, upper bound: 7.4194108
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.4194101, upper bound: 7.4194108
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6277781, upper bound: 7.6277798
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6277781, upper bound: 7.6277798
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309302
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309302
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309302
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 7.05
Output dim: 2, lower bound: -7.6309304, upper bound: 7.6309301

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
time: 2.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289835, upper bound: 7.6289829
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289831
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
time: 1.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287047, upper bound: 7.6287046
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287047, upper bound: 7.6287046
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276678, upper bound: 7.6276675
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276680, upper bound: 7.6276673
time: 2.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5715201, upper bound: 7.5715212
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5715211, upper bound: 7.5715195
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315886, upper bound: 7.6315883
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315886, upper bound: 7.6315883
time: 2.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 44

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309782, upper bound: 7.6309781
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309783, upper bound: 7.6309777
time: 3.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324319
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324318
time: 6.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324315, upper bound: 7.6324312
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324315, upper bound: 7.6324318
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324166, upper bound: 7.6324160
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324166, upper bound: 7.6324165
time: 2.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324166
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324159
time: 4.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4166408, upper bound: 7.4166417
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4166408, upper bound: 7.4166417
time: 3.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302014, upper bound: 7.6302020
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302014, upper bound: 7.6302020
time: 2.08 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310128, upper bound: 7.6310127
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310128, upper bound: 7.6310133
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309991, upper bound: 7.6309994
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309991, upper bound: 7.6309992
time: 2.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318525, upper bound: 7.6318524
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318525, upper bound: 7.6318530
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 96

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316350, upper bound: 7.6316352
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316350, upper bound: 7.6316356
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314342, upper bound: 7.6314348
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314342, upper bound: 7.6314348
time: 4.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311642, upper bound: 7.6311642
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311642, upper bound: 7.6311642
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5811047, upper bound: 7.5811093
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5811047, upper bound: 7.5811083
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5822422, upper bound: 7.5822442
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5822422, upper bound: 7.5822441
time: 4.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308760, upper bound: 7.6308762
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308764, upper bound: 7.6308758
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
time: 2.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309302
time: 3.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
time: 4.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6243033, upper bound: 7.6243039
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6243033, upper bound: 7.6243039
time: 2.18 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.23 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289835, upper bound: 7.6289829
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289834, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6287047, upper bound: 7.6287046
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6287047, upper bound: 7.6287046
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6276678, upper bound: 7.6276675
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6276680, upper bound: 7.6276673
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.5715201, upper bound: 7.5715212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.5715211, upper bound: 7.5715195
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6315886, upper bound: 7.6315883
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6315886, upper bound: 7.6315883
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309782, upper bound: 7.6309781
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309783, upper bound: 7.6309777
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324319
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324318
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324315, upper bound: 7.6324312
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324315, upper bound: 7.6324318
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324166, upper bound: 7.6324160
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324166, upper bound: 7.6324165
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324166
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324159
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.4166408, upper bound: 7.4166417
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.4166408, upper bound: 7.4166417
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6302014, upper bound: 7.6302020
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6302014, upper bound: 7.6302020
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6310128, upper bound: 7.6310127
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6310128, upper bound: 7.6310133
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309991, upper bound: 7.6309994
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309991, upper bound: 7.6309992
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6318525, upper bound: 7.6318524
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6318525, upper bound: 7.6318530
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6316350, upper bound: 7.6316352
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6316350, upper bound: 7.6316356
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6314342, upper bound: 7.6314348
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6314342, upper bound: 7.6314348
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6311642, upper bound: 7.6311642
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6311642, upper bound: 7.6311642
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.5811047, upper bound: 7.5811093
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.5811047, upper bound: 7.5811083
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.5822422, upper bound: 7.5822442
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.5822422, upper bound: 7.5822441
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6308760, upper bound: 7.6308762
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6308764, upper bound: 7.6308758
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309302
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6243033, upper bound: 7.6243039
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 5.23
Output dim: 2, lower bound: -7.6243033, upper bound: 7.6243039

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289830
time: 1.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287046, upper bound: 7.6287046
time: 2.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287046, upper bound: 7.6287047
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287317, upper bound: 7.6287316
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287317, upper bound: 7.6287315
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289657, upper bound: 7.6289657
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289660, upper bound: 7.6289656
time: 3.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
time: 2.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 230

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289657, upper bound: 7.6289658
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289659, upper bound: 7.6289656
time: 2.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3661657, upper bound: 7.3661652
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3661657, upper bound: 7.3661652
time: 2.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283466, upper bound: 7.6283463
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283466, upper bound: 7.6283463
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276676, upper bound: 7.6276672
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276677, upper bound: 7.6276673
time: 4.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 247

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6161857, upper bound: 7.6161848
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6161857, upper bound: 7.6161860
time: 2.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
time: 3.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5679002, upper bound: 7.5678983
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5679002, upper bound: 7.5678973
time: 2.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287318, upper bound: 7.6287315
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287318, upper bound: 7.6287315
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4580394, upper bound: 7.4580391
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4580394, upper bound: 7.4580391
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315884, upper bound: 7.6315884
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315884, upper bound: 7.6315883
time: 2.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307869, upper bound: 7.6307873
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307869, upper bound: 7.6307873
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308954, upper bound: 7.6308952
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308954, upper bound: 7.6308952
time: 2.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324313, upper bound: 7.6324312
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324319
time: 2.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295858, upper bound: 7.6295856
time: 2.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295858, upper bound: 7.6295856
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321387, upper bound: 7.6321385
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6321387, upper bound: 7.6321390
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291697
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291697
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324159
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324160
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291700
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291700
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324161, upper bound: 7.6324165
time: 2.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324162, upper bound: 7.6324160
time: 3.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324161, upper bound: 7.6324165
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6324161, upper bound: 7.6324165
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283112, upper bound: 7.6283122
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6283112, upper bound: 7.6283113
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 58

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302012, upper bound: 7.6302015
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302018, upper bound: 7.6302010
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303885, upper bound: 7.6303888
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303887, upper bound: 7.6303886
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310126, upper bound: 7.6310135
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310131, upper bound: 7.6310133
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309985, upper bound: 7.6309993
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309985, upper bound: 7.6309993
time: 10.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303847, upper bound: 7.6303850
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303849, upper bound: 7.6303848
time: 3.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318523, upper bound: 7.6318529
time: 2.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6318523, upper bound: 7.6318529
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315113, upper bound: 7.6315114
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315116, upper bound: 7.6315119
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -5.1286879, 3.9331427, -5.1286879, 3.9331427, -9.0618305, 9.0618305
1: -4.0091529, 3.6131139, -4.0091529, 3.6131139, -7.6222668, 7.6222668
2: -6.6639881, 2.7243943, -6.6639881, 2.7243943, -9.3883820, 9.3883820
3: -5.8328662, 2.9040592, -5.8328662, 2.9040592, -8.7369251, 8.7369251
4: -6.1130657, 3.8349376, -6.1130657, 3.8349376, -9.9480038, 9.9480038
5: -4.6064701, 4.0478497, -4.6064701, 4.0478497, -8.6543198, 8.6543198
6: -4.8427486, 4.1230493, -4.8427486, 4.1230493, -8.9657974, 8.9657974
7: -5.5768681, 4.1297841, -5.5768681, 4.1297841, -9.7066517, 9.7066517
8: -6.3826962, 3.8029628, -6.3826962, 3.8029628, -10.1856594, 10.1856594
9: -4.5053129, 5.1798744, -4.5053129, 5.1798744, -9.6851873, 9.6851873

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316350, upper bound: 7.6316356
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316351, upper bound: 7.6316349
time: 4.81 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 8.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289833
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289830
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6287046, upper bound: 7.6287046
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6287046, upper bound: 7.6287047
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6287317, upper bound: 7.6287316
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6287317, upper bound: 7.6287315
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289657, upper bound: 7.6289657
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289660, upper bound: 7.6289656
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289657, upper bound: 7.6289658
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289659, upper bound: 7.6289656
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.3661657, upper bound: 7.3661652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.3661657, upper bound: 7.3661652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6283466, upper bound: 7.6283463
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6283466, upper bound: 7.6283463
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6276676, upper bound: 7.6276672
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6276677, upper bound: 7.6276673
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6161857, upper bound: 7.6161848
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6161857, upper bound: 7.6161860
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6284556, upper bound: 7.6284554
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6289832, upper bound: 7.6289831
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.5679002, upper bound: 7.5678983
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.5679002, upper bound: 7.5678973
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6287318, upper bound: 7.6287315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6287318, upper bound: 7.6287315
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.4580394, upper bound: 7.4580391
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.4580394, upper bound: 7.4580391
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6315884, upper bound: 7.6315884
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6315884, upper bound: 7.6315883
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6307869, upper bound: 7.6307873
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6307869, upper bound: 7.6307873
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6308954, upper bound: 7.6308952
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6308954, upper bound: 7.6308952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324313, upper bound: 7.6324312
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324314, upper bound: 7.6324319
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6295858, upper bound: 7.6295856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6295858, upper bound: 7.6295856
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6321387, upper bound: 7.6321385
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6321387, upper bound: 7.6321390
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291697
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291697
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324159
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324167, upper bound: 7.6324160
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6291702, upper bound: 7.6291700
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324161, upper bound: 7.6324165
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324162, upper bound: 7.6324160
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324161, upper bound: 7.6324165
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6324161, upper bound: 7.6324165
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6283112, upper bound: 7.6283122
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6283112, upper bound: 7.6283113
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6302012, upper bound: 7.6302015
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6302018, upper bound: 7.6302010
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6303885, upper bound: 7.6303888
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6303887, upper bound: 7.6303886
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6310126, upper bound: 7.6310135
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6310131, upper bound: 7.6310133
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6309985, upper bound: 7.6309993
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6309985, upper bound: 7.6309993
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6303847, upper bound: 7.6303850
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6303849, upper bound: 7.6303848
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6318523, upper bound: 7.6318529
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6318523, upper bound: 7.6318529
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6315113, upper bound: 7.6315114
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6315116, upper bound: 7.6315119
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6316350, upper bound: 7.6316356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 8.00
Output dim: 2, lower bound: -7.6316351, upper bound: 7.6316349
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6316350, upper bound: 7.6316356
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6314342, upper bound: 7.6314348
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6314342, upper bound: 7.6314348
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6311642, upper bound: 7.6311642
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6311642, upper bound: 7.6311642
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6308760, upper bound: 7.6308762
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6308764, upper bound: 7.6308758
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309302
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 8.00
Output dim: 2, lower bound: -7.6309303, upper bound: 7.6309301

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 5.60 + 595.75 = 601.35 seconds
