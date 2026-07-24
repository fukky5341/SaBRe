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
execution time: IAR + RelationalAnalysis = 0.97 + 4.97 = 5.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6335958, upper bound: 7.6335962

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335580, upper bound: 7.6335586
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335580, upper bound: 7.6335581
time: 2.26 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.77 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.77
Output dim: 2, lower bound: -7.6335580, upper bound: 7.6335586
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.77
Output dim: 2, lower bound: -7.6335580, upper bound: 7.6335581

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

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332658, upper bound: 7.6332661
time: 9.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332658, upper bound: 7.6332661
time: 2.44 seconds

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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332657, upper bound: 7.6332657
time: 2.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332657, upper bound: 7.6332657
time: 2.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 6.42 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.42
Output dim: 2, lower bound: -7.6332658, upper bound: 7.6332661
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.42
Output dim: 2, lower bound: -7.6332658, upper bound: 7.6332661
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 6.42
Output dim: 2, lower bound: -7.6332657, upper bound: 7.6332657
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 6.42
Output dim: 2, lower bound: -7.6332657, upper bound: 7.6332657

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311929
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311930
time: 3.11 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311923, upper bound: 7.6311926
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311923, upper bound: 7.6311925
time: 1.74 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311924
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311924
time: 1.92 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311924, upper bound: 7.6311919
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311924, upper bound: 7.6311918
time: 3.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 7.27 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311929
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311930
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311923, upper bound: 7.6311926
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311923, upper bound: 7.6311925
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311924
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311920, upper bound: 7.6311924
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311924, upper bound: 7.6311919
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 7.27
Output dim: 2, lower bound: -7.6311924, upper bound: 7.6311918

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 1.95 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 1.97 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310169
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 2.06 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310169
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 1.95 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 2.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 3.07 seconds

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 3.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310168
time: 3.87 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310169
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 2.37 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
time: 2.39 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.42 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310169
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310169
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310168
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310169
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.42
Output dim: 2, lower bound: -7.6310163, upper bound: 7.6310163

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179338
time: 2.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179332
time: 2.73 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179342
time: 9.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179341
time: 2.57 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179332
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179332
time: 2.37 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179344
time: 10.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179341
time: 1.86 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179339
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179341
time: 4.09 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179339
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179345
time: 2.54 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179342
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179334
time: 2.50 seconds

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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179342
time: 4.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179343
time: 2.49 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
time: 1.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
time: 2.16 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
time: 2.06 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
time: 2.13 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179340
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
time: 1.49 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179340
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179340
time: 1.66 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
time: 1.55 seconds

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
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 247
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 58
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 65

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
time: 4.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
time: 5.37 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 10.32 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179338
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179332
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179342
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179341
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179332
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179332
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179344
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179341
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179339
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179341
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179339
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179345
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179342
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179334
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179342
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179340, upper bound: 7.6179343
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179342
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179340
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179340
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179340
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179337
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 10.32
Output dim: 2, lower bound: -7.6179337, upper bound: 7.6179338

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 5.95 + 197.68 = 203.63 seconds
