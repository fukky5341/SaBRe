## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 7.6259627037
Search space: {k/256 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 1.49 + 5.71 = 7.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6336112, upper bound: 7.6336109


# Binary Search by BASE starts (time budget: 1992.80 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=9.388381958007812
rel_dist={2: [-7.633606580437654, 7.633606672931759]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=9.388381958007812
rel_dist={2: [-7.633595790761577, 7.633596173769259]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=9.388381958007812
rel_dist={2: [-7.63358222082411, 7.633582568333054]}

## Binary Search Result
Binary search time: 18.83 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1973.97 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335696, upper bound: 7.6335699
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335699, upper bound: 7.6335694
time: 2.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.48 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.48
Output dim: 2, lower bound: -7.6335696, upper bound: 7.6335699
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.48
Output dim: 2, lower bound: -7.6335699, upper bound: 7.6335694

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332777, upper bound: 7.6332777
time: 2.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332777, upper bound: 7.6332771
time: 2.37 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332777, upper bound: 7.6332770
time: 9.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332778, upper bound: 7.6332775
time: 3.27 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -7.6332777, upper bound: 7.6332777
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -7.6332777, upper bound: 7.6332771
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -7.6332777, upper bound: 7.6332770
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.73
Output dim: 2, lower bound: -7.6332778, upper bound: 7.6332775

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312058
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312056
time: 3.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312056
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312057
time: 2.29 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051
time: 3.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051
time: 2.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 7.44 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312058
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312056
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312056
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312045, upper bound: 7.6312057
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.44
Output dim: 2, lower bound: -7.6312051, upper bound: 7.6312051

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.49 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310289
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310284
time: 3.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310285
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310285
time: 2.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310289
time: 1.98 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310289
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310284
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310285
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310285
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310290
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.83
Output dim: 2, lower bound: -7.6310284, upper bound: 7.6310289

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 5.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179476
time: 3.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179476
time: 3.09 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179486
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179488
time: 2.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179487
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 3.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
time: 2.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179492
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179485
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179497
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179495
time: 2.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179495
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179496
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179494
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179494
time: 2.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179491
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179493
time: 2.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179493
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179493
time: 2.48 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179494
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179497
time: 2.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179485
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179491
time: 2.84 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.53 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179476
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179476
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179486
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179488
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179487
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179489, upper bound: 7.6179478
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179492
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179485
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179497
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179495
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179496
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179494
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179494
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179491
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179493
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179494
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179497
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179485
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.53
Output dim: 2, lower bound: -7.6179481, upper bound: 7.6179491
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=9.388381958007812
rel_dist={2: [-7.633606580437654, 7.633606672931759]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335721, upper bound: 7.6335724
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335721, upper bound: 7.6335720
time: 3.31 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.23 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.23
Output dim: 2, lower bound: -7.6335721, upper bound: 7.6335724
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.23
Output dim: 2, lower bound: -7.6335721, upper bound: 7.6335720

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332794
time: 1.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332786
time: 1.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332791
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332794
time: 2.40 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.95 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.95
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332794
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.95
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332786
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.95
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332791
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.95
Output dim: 2, lower bound: -7.6332787, upper bound: 7.6332794

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312077
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312077
time: 2.27 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312068
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312069
time: 2.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312077
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312078
time: 3.47 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312069
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312069
time: 3.87 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 8.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312077
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312077
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312068
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312069
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312077
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312063, upper bound: 7.6312078
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312069
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.28
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312069

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310297
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310297
time: 1.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310302
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310302
time: 2.26 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310296
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.32 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310301
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310296
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
time: 2.11 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.71 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310297
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310297
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310302
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310297, upper bound: 7.6310302
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310296
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310301
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310296
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.71
Output dim: 2, lower bound: -7.6310296, upper bound: 7.6310302

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179547
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179543
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179548
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179537
time: 2.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179543
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179548
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179537
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179545
time: 3.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179549
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179537
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179546
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179549
time: 2.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
time: 2.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
time: 2.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
time: 2.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179562
time: 2.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179558
time: 7.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179558
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
time: 2.45 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179558
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
time: 3.43 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179547
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179543
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179548
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179537
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179543
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179548
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179537
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179545
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179549
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179537
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179546
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179544
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179552, upper bound: 7.6179549
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179562
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179558
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179559
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179558
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179558
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.78
Output dim: 2, lower bound: -7.6179540, upper bound: 7.6179560
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=9.388381958007812
rel_dist={2: [-7.633608742420405, 7.633608999615372]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335737, upper bound: 7.6335737
time: 1.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335737, upper bound: 7.6335740
time: 2.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.66 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.66
Output dim: 2, lower bound: -7.6335737, upper bound: 7.6335737
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.66
Output dim: 2, lower bound: -7.6335737, upper bound: 7.6335740

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332796, upper bound: 7.6332805
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332796, upper bound: 7.6332802
time: 2.80 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332797, upper bound: 7.6332805
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332797, upper bound: 7.6332804
time: 1.90 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.62 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 2, lower bound: -7.6332796, upper bound: 7.6332805
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 2, lower bound: -7.6332796, upper bound: 7.6332802
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 2, lower bound: -7.6332797, upper bound: 7.6332805
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.62
Output dim: 2, lower bound: -7.6332797, upper bound: 7.6332804

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312085
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312087
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312076
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312076
time: 2.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312087
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312086
time: 1.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312076
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312071
time: 2.41 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312085
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312070, upper bound: 7.6312087
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312076
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312076
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312087
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312071, upper bound: 7.6312086
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312076
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.29
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312071

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310305
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 3.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310307
time: 2.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310305
time: 2.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310307
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310307
time: 2.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 3.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 2.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 2.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310305
time: 2.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 2.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
time: 2.21 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310305
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310307
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310305
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310307
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310307
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310305
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.93
Output dim: 2, lower bound: -7.6310300, upper bound: 7.6310306

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179554
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179561
time: 2.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179556
time: 1.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179563
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179563
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179556
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179563
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179559
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179560
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179561
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
time: 1.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179553
time: 2.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
time: 1.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179568
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179566
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179575
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179570
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
time: 5.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179573
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
time: 3.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179575
time: 6.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179576
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179565
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179566
time: 6.35 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 10.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179554
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179561
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179556
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179563
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179563
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179556
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179563
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179560
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179561
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179553
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179569, upper bound: 7.6179562
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179568
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179566
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179575
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179570
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179573
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179575
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179576
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179565
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179574
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 10.18
Output dim: 2, lower bound: -7.6179554, upper bound: 7.6179566
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=9.388381958007812
rel_dist={2: [-7.633610057465512, 7.633610415048498]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335746, upper bound: 7.6335746
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335747, upper bound: 7.6335748
time: 1.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.55 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.55
Output dim: 2, lower bound: -7.6335746, upper bound: 7.6335746
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.55
Output dim: 2, lower bound: -7.6335747, upper bound: 7.6335748

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332804, upper bound: 7.6332811
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332804, upper bound: 7.6332803
time: 1.74 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332803, upper bound: 7.6332810
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6332803, upper bound: 7.6332810
time: 1.80 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 2, lower bound: -7.6332804, upper bound: 7.6332811
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 2, lower bound: -7.6332804, upper bound: 7.6332803
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 2, lower bound: -7.6332803, upper bound: 7.6332810
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.94
Output dim: 2, lower bound: -7.6332803, upper bound: 7.6332810

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312090
time: 1.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312084
time: 2.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312079
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312079
time: 2.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312084
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312083
time: 2.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312080
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312074
time: 2.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.37 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312090
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312084
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312079
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312079
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312084
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312079, upper bound: 7.6312083
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312080
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.37
Output dim: 2, lower bound: -7.6312090, upper bound: 7.6312074

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310303
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 2.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 2.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 2.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310302
time: 2.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310303
time: 3.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
time: 2.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
time: 2.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310303
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310302
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310303
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310308
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.66
Output dim: 2, lower bound: -7.6310308, upper bound: 7.6310309

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179557
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179553
time: 2.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179559
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179559
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
time: 1.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179553
time: 2.24 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179570
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
time: 1.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179567
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179557
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
time: 1.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179566
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179570
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
time: 1.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179557
time: 1.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179584
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179583
time: 2.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179585
time: 2.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179580
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179581
time: 2.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179574
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179582
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179580
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179586
time: 2.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179574
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179583
time: 4.01 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179557
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179553
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179559
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179559
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179553
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179570
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179567
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179557
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179566
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179570
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179568
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179587, upper bound: 7.6179557
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179584
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179583
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179585
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179579
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179580
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179581
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179574
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179582
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179580
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179586
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179574
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.03
Output dim: 2, lower bound: -7.6179571, upper bound: 7.6179583
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=9.388381958007812
rel_dist={2: [-7.6336112072082445, 7.633610906854749]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 841.96 seconds
