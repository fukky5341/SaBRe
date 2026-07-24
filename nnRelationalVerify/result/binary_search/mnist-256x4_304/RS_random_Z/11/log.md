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
execution time: IAR + LP analysis = 1.45 + 5.67 = 7.12 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -7.6336112, upper bound: 7.6336109


# Binary Search by BASE starts (time budget: 1992.88 seconds, max iter: 100)

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
Binary search time: 18.73 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1974.15 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335696, upper bound: 7.6335699
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335699, upper bound: 7.6335694
time: 2.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.30 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.30
Output dim: 2, lower bound: -7.6335696, upper bound: 7.6335699
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.30
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335698, upper bound: 7.6335698
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335698, upper bound: 7.6335698
time: 3.32 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311719, upper bound: 7.6311726
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311719, upper bound: 7.6311725
time: 2.25 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.14 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.14
Output dim: 2, lower bound: -7.6335698, upper bound: 7.6335698
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.14
Output dim: 2, lower bound: -7.6335698, upper bound: 7.6335698
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.14
Output dim: 2, lower bound: -7.6311719, upper bound: 7.6311726
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.14
Output dim: 2, lower bound: -7.6311719, upper bound: 7.6311725

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316750, upper bound: 7.6316757
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316750, upper bound: 7.6316756
time: 2.11 seconds

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335696, upper bound: 7.6335697
time: 3.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335694, upper bound: 7.6335694
time: 3.61 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308962, upper bound: 7.6308974
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308963, upper bound: 7.6308975
time: 5.43 seconds

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308962, upper bound: 7.6308974
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308963, upper bound: 7.6308969
time: 4.16 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 7.47 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6316750, upper bound: 7.6316757
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6316750, upper bound: 7.6316756
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6335696, upper bound: 7.6335697
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6335694, upper bound: 7.6335694
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6308962, upper bound: 7.6308974
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6308963, upper bound: 7.6308975
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6308962, upper bound: 7.6308974
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.47
Output dim: 2, lower bound: -7.6308963, upper bound: 7.6308969

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310086
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310092
time: 2.26 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316307, upper bound: 7.6316310
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6316307, upper bound: 7.6316311
time: 2.05 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314425, upper bound: 7.6314421
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314425, upper bound: 7.6314422
time: 2.24 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335688, upper bound: 7.6335694
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6335691, upper bound: 7.6335689
time: 3.28 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303892, upper bound: 7.6303896
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303892, upper bound: 7.6303896
time: 1.99 seconds

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308969, upper bound: 7.6308975
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308968, upper bound: 7.6308975
time: 2.18 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5763902, upper bound: 7.5763904
time: 2.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5763902, upper bound: 7.5763904
time: 2.44 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5219912, upper bound: 7.5219915
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5219912, upper bound: 7.5219915
time: 2.32 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.97 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310086
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310092
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6316307, upper bound: 7.6316310
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6316307, upper bound: 7.6316311
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6314425, upper bound: 7.6314421
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6314425, upper bound: 7.6314422
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6335688, upper bound: 7.6335694
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6335691, upper bound: 7.6335689
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6303892, upper bound: 7.6303896
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6303892, upper bound: 7.6303896
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6308969, upper bound: 7.6308975
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.6308968, upper bound: 7.6308975
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.5763902, upper bound: 7.5763904
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.5763902, upper bound: 7.5763904
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.5219912, upper bound: 7.5219915
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.97
Output dim: 2, lower bound: -7.5219912, upper bound: 7.5219915

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310087, upper bound: 7.6310092
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310092
time: 5.40 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310092, upper bound: 7.6310086
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310092, upper bound: 7.6310092
time: 2.03 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315593, upper bound: 7.6315593
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315593, upper bound: 7.6315587
time: 2.82 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309770, upper bound: 7.6309773
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309770, upper bound: 7.6309773
time: 2.00 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312048, upper bound: 7.6312058
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312053, upper bound: 7.6312051
time: 2.31 seconds

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280569
time: 1.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280569
time: 1.42 seconds

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5552452, upper bound: 7.5552448
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5552452, upper bound: 7.5552443
time: 3.69 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315019, upper bound: 7.6315009
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315019, upper bound: 7.6315005
time: 2.36 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292947
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292947
time: 2.04 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292946
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292946
time: 3.02 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763907
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763908
time: 2.62 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308962, upper bound: 7.6308975
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308964, upper bound: 7.6308963
time: 3.86 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6310087, upper bound: 7.6310092
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6310092, upper bound: 7.6310086
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6310092, upper bound: 7.6310092
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6315593, upper bound: 7.6315593
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6315593, upper bound: 7.6315587
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6309770, upper bound: 7.6309773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6309770, upper bound: 7.6309773
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6312048, upper bound: 7.6312058
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6312053, upper bound: 7.6312051
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280569
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280569
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.5552452, upper bound: 7.5552448
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.5552452, upper bound: 7.5552443
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6315019, upper bound: 7.6315009
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6315019, upper bound: 7.6315005
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292947
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292947
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292946
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292946
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763907
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763908
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6308962, upper bound: 7.6308975
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 7.50
Output dim: 2, lower bound: -7.6308964, upper bound: 7.6308963

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310044, upper bound: 7.6310041
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310044, upper bound: 7.6310045
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310086
time: 1.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310091
time: 1.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310083, upper bound: 7.6310087
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310088
time: 2.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310092
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310085, upper bound: 7.6310092
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276099, upper bound: 7.6276113
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6276099, upper bound: 7.6276113
time: 2.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315593, upper bound: 7.6315593
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315594, upper bound: 7.6315592
time: 2.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302059, upper bound: 7.6302058
time: 2.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302059, upper bound: 7.6302059
time: 2.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309769, upper bound: 7.6309773
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309770, upper bound: 7.6309773
time: 1.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5190055, upper bound: 7.5190048
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5190055, upper bound: 7.5190048
time: 2.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6296380, upper bound: 7.6296384
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6296380, upper bound: 7.6296384
time: 2.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280555, upper bound: 7.6280584
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280562, upper bound: 7.6280570
time: 2.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280562, upper bound: 7.6280559
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280560, upper bound: 7.6280573
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5220091, upper bound: 7.5220081
time: 1.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5220091, upper bound: 7.5220081
time: 1.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308109, upper bound: 7.6308107
time: 3.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308109, upper bound: 7.6308108
time: 2.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287711, upper bound: 7.6287715
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287711, upper bound: 7.6287714
time: 3.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292937, upper bound: 7.6292946
time: 2.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292943
time: 3.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6282826, upper bound: 7.6282835
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6282831, upper bound: 7.6282825
time: 2.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277543, upper bound: 7.6277550
time: 2.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277549, upper bound: 7.6277535
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6301532, upper bound: 7.6301537
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6301531, upper bound: 7.6301537
time: 2.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308960, upper bound: 7.6308963
time: 7.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6308963, upper bound: 7.6308966
time: 2.37 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 11.18 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310044, upper bound: 7.6310041
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310044, upper bound: 7.6310045
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310086
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310091
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310083, upper bound: 7.6310087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310088
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310086, upper bound: 7.6310092
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6310085, upper bound: 7.6310092
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6276099, upper bound: 7.6276113
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6276099, upper bound: 7.6276113
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6315593, upper bound: 7.6315593
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6315594, upper bound: 7.6315592
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6302059, upper bound: 7.6302058
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6302059, upper bound: 7.6302059
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6309769, upper bound: 7.6309773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6309770, upper bound: 7.6309773
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.5190055, upper bound: 7.5190048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.5190055, upper bound: 7.5190048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6296380, upper bound: 7.6296384
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6296380, upper bound: 7.6296384
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6280555, upper bound: 7.6280584
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6280562, upper bound: 7.6280570
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6280562, upper bound: 7.6280559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6280560, upper bound: 7.6280573
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.5220091, upper bound: 7.5220081
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.5220091, upper bound: 7.5220081
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6308109, upper bound: 7.6308107
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6308109, upper bound: 7.6308108
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6287711, upper bound: 7.6287715
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6287711, upper bound: 7.6287714
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6292937, upper bound: 7.6292946
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292943
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6282826, upper bound: 7.6282835
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6282831, upper bound: 7.6282825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6277543, upper bound: 7.6277550
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6277549, upper bound: 7.6277535
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6301532, upper bound: 7.6301537
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6301531, upper bound: 7.6301537
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6308960, upper bound: 7.6308963
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 11.18
Output dim: 2, lower bound: -7.6308963, upper bound: 7.6308966

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284122, upper bound: 7.6284119
time: 1.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284122, upper bound: 7.6284121
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309909, upper bound: 7.6309913
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309912, upper bound: 7.6309911
time: 3.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310045, upper bound: 7.6310045
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310045, upper bound: 7.6310045
time: 2.39 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2975906, upper bound: 7.2975900
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2975906, upper bound: 7.2975900
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284126, upper bound: 7.6284117
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284126, upper bound: 7.6284121
time: 2.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309290, upper bound: 7.6309294
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309290, upper bound: 7.6309294
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309978, upper bound: 7.6309976
time: 3.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309980, upper bound: 7.6309979
time: 1.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310091, upper bound: 7.6310087
time: 10.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6310091, upper bound: 7.6310087
time: 2.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6272004, upper bound: 7.6272002
time: 1.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6272004, upper bound: 7.6272004
time: 2.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6273730, upper bound: 7.6273746
time: 1.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6273747, upper bound: 7.6273741
time: 1.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286983, upper bound: 7.6286982
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6286983, upper bound: 7.6286979
time: 3.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315585, upper bound: 7.6315587
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315589, upper bound: 7.6315589
time: 2.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6296914, upper bound: 7.6296920
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6296914, upper bound: 7.6296920
time: 2.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302055, upper bound: 7.6302059
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6302059, upper bound: 7.6302055
time: 2.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304403, upper bound: 7.6304409
time: 1.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304408, upper bound: 7.6304406
time: 1.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309310, upper bound: 7.6309314
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309310, upper bound: 7.6309314
time: 3.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6296379, upper bound: 7.6296384
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6296379, upper bound: 7.6296384
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1739903, upper bound: 7.1739904
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1739903, upper bound: 7.1739904
time: 2.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5183441, upper bound: 7.5183467
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5183442, upper bound: 7.5183461
time: 1.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280567
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280567
time: 1.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279410, upper bound: 7.6279409
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6279406, upper bound: 7.6279406
time: 1.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280557, upper bound: 7.6280573
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6280560, upper bound: 7.6280561
time: 2.46 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307385, upper bound: 7.6307386
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307396, upper bound: 7.6307375
time: 2.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305595, upper bound: 7.6305591
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305595, upper bound: 7.6305584
time: 2.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287711, upper bound: 7.6287715
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287712, upper bound: 7.6287713
time: 2.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6274557, upper bound: 7.6274578
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6274575, upper bound: 7.6274566
time: 4.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277531, upper bound: 7.6277550
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277537, upper bound: 7.6277546
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292942
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6292937, upper bound: 7.6292944
time: 2.38 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2068015, upper bound: 7.2068040
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2068015, upper bound: 7.2068040
time: 2.19 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6282827, upper bound: 7.6282825
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6282815, upper bound: 7.6282825
time: 3.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277541, upper bound: 7.6277544
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6277534, upper bound: 7.6277555
time: 2.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5047042, upper bound: 7.5047042
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5047053, upper bound: 7.5047035
time: 1.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295245, upper bound: 7.6295253
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295250, upper bound: 7.6295243
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5669787, upper bound: 7.5669782
time: 2.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5669787, upper bound: 7.5669780
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763863
time: 1.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763863
time: 1.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306023, upper bound: 7.6306011
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306023, upper bound: 7.6306017
time: 1.86 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.88 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6284122, upper bound: 7.6284119
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6284122, upper bound: 7.6284121
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309909, upper bound: 7.6309913
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309912, upper bound: 7.6309911
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6310045, upper bound: 7.6310045
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6310045, upper bound: 7.6310045
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.2975906, upper bound: 7.2975900
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.2975906, upper bound: 7.2975900
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6284126, upper bound: 7.6284117
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6284126, upper bound: 7.6284121
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309290, upper bound: 7.6309294
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309290, upper bound: 7.6309294
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309978, upper bound: 7.6309976
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309980, upper bound: 7.6309979
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6310091, upper bound: 7.6310087
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6310091, upper bound: 7.6310087
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6272004, upper bound: 7.6272002
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6272004, upper bound: 7.6272004
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6273730, upper bound: 7.6273746
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6273747, upper bound: 7.6273741
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6286983, upper bound: 7.6286982
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6286983, upper bound: 7.6286979
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6315585, upper bound: 7.6315587
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6315589, upper bound: 7.6315589
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6296914, upper bound: 7.6296920
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6296914, upper bound: 7.6296920
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6302055, upper bound: 7.6302059
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6302059, upper bound: 7.6302055
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6304403, upper bound: 7.6304409
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6304408, upper bound: 7.6304406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309310, upper bound: 7.6309314
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6309310, upper bound: 7.6309314
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6296379, upper bound: 7.6296384
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6296379, upper bound: 7.6296384
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.1739903, upper bound: 7.1739904
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.1739903, upper bound: 7.1739904
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5183441, upper bound: 7.5183467
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5183442, upper bound: 7.5183461
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280567
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6280571, upper bound: 7.6280567
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6279410, upper bound: 7.6279409
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6279406, upper bound: 7.6279406
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6280557, upper bound: 7.6280573
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6280560, upper bound: 7.6280561
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6307385, upper bound: 7.6307386
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6307396, upper bound: 7.6307375
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6305595, upper bound: 7.6305591
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6305595, upper bound: 7.6305584
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6287711, upper bound: 7.6287715
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6287712, upper bound: 7.6287713
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6274557, upper bound: 7.6274578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6274575, upper bound: 7.6274566
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6277531, upper bound: 7.6277550
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6277537, upper bound: 7.6277546
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6292939, upper bound: 7.6292942
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6292937, upper bound: 7.6292944
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.2068015, upper bound: 7.2068040
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.2068015, upper bound: 7.2068040
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6282827, upper bound: 7.6282825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6282815, upper bound: 7.6282825
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6277541, upper bound: 7.6277544
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6277534, upper bound: 7.6277555
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5047042, upper bound: 7.5047042
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5047053, upper bound: 7.5047035
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6295245, upper bound: 7.6295253
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6295250, upper bound: 7.6295243
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5669787, upper bound: 7.5669782
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5669787, upper bound: 7.5669780
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.5763885, upper bound: 7.5763863
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6306023, upper bound: 7.6306011
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.88
Output dim: 2, lower bound: -7.6306023, upper bound: 7.6306017

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271491, upper bound: 7.6271516
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6271506, upper bound: 7.6271502
time: 3.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284114, upper bound: 7.6284122
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6284123, upper bound: 7.6284110
time: 1.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4721395, upper bound: 7.4721411
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4721395, upper bound: 7.4721411
time: 2.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309336, upper bound: 7.6309341
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309338, upper bound: 7.6309340
time: 2.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305561, upper bound: 7.6305561
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305561, upper bound: 7.6305554
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=9.388381958007812
rel_dist={2: [-7.633606580437654, 7.633606672931759]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315190, upper bound: 7.6315185
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315190, upper bound: 7.6315185
time: 1.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.73
Output dim: 2, lower bound: -7.6315190, upper bound: 7.6315185
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.73
Output dim: 2, lower bound: -7.6315190, upper bound: 7.6315185

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312122, upper bound: 7.6312119
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312122, upper bound: 7.6312125
time: 4.06 seconds

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315189, upper bound: 7.6315185
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315191, upper bound: 7.6315189
time: 2.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.57 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.57
Output dim: 2, lower bound: -7.6312122, upper bound: 7.6312119
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.57
Output dim: 2, lower bound: -7.6312122, upper bound: 7.6312125
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.57
Output dim: 2, lower bound: -7.6315189, upper bound: 7.6315185
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.57
Output dim: 2, lower bound: -7.6315191, upper bound: 7.6315189

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312121
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312119
time: 2.28 seconds

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309717, upper bound: 7.6309723
time: 3.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309717, upper bound: 7.6309717
time: 2.54 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311958, upper bound: 7.6311963
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311958, upper bound: 7.6311964
time: 2.58 seconds

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315159, upper bound: 7.6315163
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315159, upper bound: 7.6315157
time: 2.74 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.67 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312121
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312119
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6309717, upper bound: 7.6309723
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6309717, upper bound: 7.6309717
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6311958, upper bound: 7.6311963
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6311958, upper bound: 7.6311964
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6315159, upper bound: 7.6315163
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.67
Output dim: 2, lower bound: -7.6315159, upper bound: 7.6315157

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311442, upper bound: 7.6311449
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311443, upper bound: 7.6311446
time: 2.68 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312115, upper bound: 7.6312125
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312116, upper bound: 7.6312124
time: 2.80 seconds

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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309397, upper bound: 7.6309399
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309398, upper bound: 7.6309398
time: 2.35 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1147988, upper bound: 7.1147988
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1147988, upper bound: 7.1147993
time: 2.76 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1148072, upper bound: 7.1148075
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1148072, upper bound: 7.1148075
time: 2.05 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311963, upper bound: 7.6311964
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311963, upper bound: 7.6311958
time: 2.22 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1532363, upper bound: 7.1532371
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1532363, upper bound: 7.1532371
time: 1.91 seconds

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312618, upper bound: 7.6312617
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312618, upper bound: 7.6312617
time: 4.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 9.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6311442, upper bound: 7.6311449
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6311443, upper bound: 7.6311446
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6312115, upper bound: 7.6312125
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6312116, upper bound: 7.6312124
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6309397, upper bound: 7.6309399
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6309398, upper bound: 7.6309398
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.1147988, upper bound: 7.1147988
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.1147988, upper bound: 7.1147993
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.1148072, upper bound: 7.1148075
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.1148072, upper bound: 7.1148075
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6311963, upper bound: 7.6311964
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6311963, upper bound: 7.6311958
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.1532363, upper bound: 7.1532371
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.1532363, upper bound: 7.1532371
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6312618, upper bound: 7.6312617
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 9.28
Output dim: 2, lower bound: -7.6312618, upper bound: 7.6312617

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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305724, upper bound: 7.6305732
time: 3.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305727, upper bound: 7.6305728
time: 2.38 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2840265, upper bound: 7.2840259
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2840265, upper bound: 7.2840259
time: 1.84 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2022383, upper bound: 7.2022384
time: 1.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2022383, upper bound: 7.2022384
time: 1.67 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312124
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312125
time: 2.64 seconds

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

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309391, upper bound: 7.6309399
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309391, upper bound: 7.6309392
time: 3.12 seconds

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

Time for backsubstitution: 1.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304171
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304171
time: 2.07 seconds

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

Time for backsubstitution: 1.45 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4278407, upper bound: 7.4278411
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4278407, upper bound: 7.4278411
time: 2.14 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311957, upper bound: 7.6311963
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311957, upper bound: 7.6311963
time: 2.09 seconds

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309831, upper bound: 7.6309829
time: 3.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309830, upper bound: 7.6309824
time: 2.19 seconds

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

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312619, upper bound: 7.6312611
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312619, upper bound: 7.6312617
time: 2.97 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6305724, upper bound: 7.6305732
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6305727, upper bound: 7.6305728
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.2840265, upper bound: 7.2840259
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.2840265, upper bound: 7.2840259
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.2022383, upper bound: 7.2022384
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.2022383, upper bound: 7.2022384
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312124
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6312124, upper bound: 7.6312125
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6309391, upper bound: 7.6309399
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6309391, upper bound: 7.6309392
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304171
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304171
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.4278407, upper bound: 7.4278411
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.4278407, upper bound: 7.4278411
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6311957, upper bound: 7.6311963
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6311957, upper bound: 7.6311963
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6309831, upper bound: 7.6309829
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6309830, upper bound: 7.6309824
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6312619, upper bound: 7.6312611
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.95
Output dim: 2, lower bound: -7.6312619, upper bound: 7.6312617

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303320
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303320
time: 2.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3854193, upper bound: 7.3854192
time: 2.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3854193, upper bound: 7.3854192
time: 2.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306576, upper bound: 7.6306572
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306578, upper bound: 7.6306573
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3099970, upper bound: 7.3099968
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3099970, upper bound: 7.3099968
time: 1.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4278313, upper bound: 7.4278315
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4278313, upper bound: 7.4278314
time: 1.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309397, upper bound: 7.6309393
time: 1.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309398, upper bound: 7.6309393
time: 3.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295803, upper bound: 7.6295809
time: 2.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295803, upper bound: 7.6295809
time: 1.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304164
time: 1.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304164
time: 12.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311955, upper bound: 7.6311959
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311957, upper bound: 7.6311956
time: 2.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311382, upper bound: 7.6311384
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311385, upper bound: 7.6311387
time: 3.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309825, upper bound: 7.6309828
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309824, upper bound: 7.6309828
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309820, upper bound: 7.6309824
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309821, upper bound: 7.6309824
time: 4.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312612, upper bound: 7.6312617
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312613, upper bound: 7.6312617
time: 2.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312470, upper bound: 7.6312472
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312470, upper bound: 7.6312475
time: 2.24 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.52 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.3854193, upper bound: 7.3854192
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.3854193, upper bound: 7.3854192
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6306576, upper bound: 7.6306572
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6306578, upper bound: 7.6306573
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.3099970, upper bound: 7.3099968
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.3099970, upper bound: 7.3099968
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.4278313, upper bound: 7.4278315
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.4278313, upper bound: 7.4278314
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6309397, upper bound: 7.6309393
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6309398, upper bound: 7.6309393
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6295803, upper bound: 7.6295809
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6295803, upper bound: 7.6295809
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304164
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304164
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6311955, upper bound: 7.6311959
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6311957, upper bound: 7.6311956
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6311382, upper bound: 7.6311384
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6311385, upper bound: 7.6311387
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6309825, upper bound: 7.6309828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6309824, upper bound: 7.6309828
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6309820, upper bound: 7.6309824
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6309821, upper bound: 7.6309824
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6312612, upper bound: 7.6312617
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6312613, upper bound: 7.6312617
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6312470, upper bound: 7.6312472
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.52
Output dim: 2, lower bound: -7.6312470, upper bound: 7.6312475

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303319
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303318
time: 3.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303320
time: 9.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303317
time: 3.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306572, upper bound: 7.6306575
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306574, upper bound: 7.6306573
time: 3.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2882938, upper bound: 7.2882923
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2882938, upper bound: 7.2882923
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307212, upper bound: 7.6307221
time: 2.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307212, upper bound: 7.6307221
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304171
time: 2.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304170
time: 2.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0659434, upper bound: 7.0659434
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.0659434, upper bound: 7.0659435
time: 7.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9620905, upper bound: 6.9620903
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9620905, upper bound: 6.9620903
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303375, upper bound: 7.6303375
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303377, upper bound: 7.6303375
time: 4.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304164
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304163
time: 3.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309715, upper bound: 7.6309723
time: 5.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309715, upper bound: 7.6309717
time: 2.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309792, upper bound: 7.6309785
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309792, upper bound: 7.6309790
time: 3.71 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290868, upper bound: 7.6290878
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290868, upper bound: 7.6290878
time: 2.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4022730, upper bound: 7.4022728
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4022730, upper bound: 7.4022728
time: 2.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309818, upper bound: 7.6309826
time: 2.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309821, upper bound: 7.6309825
time: 3.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307159, upper bound: 7.6307160
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307159, upper bound: 7.6307159
time: 2.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1513030, upper bound: 7.1513027
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.1513030, upper bound: 7.1513027
time: 2.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309633, upper bound: 7.6309640
time: 3.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309633, upper bound: 7.6309640
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305847, upper bound: 7.6305849
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305848, upper bound: 7.6305845
time: 2.46 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3865833, upper bound: 7.3865818
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3865833, upper bound: 7.3865818
time: 1.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311817, upper bound: 7.6311817
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311824, upper bound: 7.6311813
time: 4.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312469, upper bound: 7.6312475
time: 2.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6312469, upper bound: 7.6312468
time: 2.48 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.99 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303319
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303318
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6303312, upper bound: 7.6303317
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6306572, upper bound: 7.6306575
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6306574, upper bound: 7.6306573
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.2882938, upper bound: 7.2882923
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.2882938, upper bound: 7.2882923
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6307212, upper bound: 7.6307221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6307212, upper bound: 7.6307221
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304171
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6304171, upper bound: 7.6304170
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.0659434, upper bound: 7.0659434
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.0659434, upper bound: 7.0659435
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -6.9620905, upper bound: 6.9620903
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -6.9620905, upper bound: 6.9620903
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6303375, upper bound: 7.6303375
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6303377, upper bound: 7.6303375
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304164
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6304166, upper bound: 7.6304163
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309715, upper bound: 7.6309723
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309715, upper bound: 7.6309717
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309792, upper bound: 7.6309785
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309792, upper bound: 7.6309790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6290868, upper bound: 7.6290878
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6290868, upper bound: 7.6290878
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.4022730, upper bound: 7.4022728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.4022730, upper bound: 7.4022728
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309818, upper bound: 7.6309826
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309821, upper bound: 7.6309825
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6307159, upper bound: 7.6307160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6307159, upper bound: 7.6307159
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.1513030, upper bound: 7.1513027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.1513030, upper bound: 7.1513027
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309633, upper bound: 7.6309640
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6309633, upper bound: 7.6309640
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6305847, upper bound: 7.6305849
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6305848, upper bound: 7.6305845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.3865833, upper bound: 7.3865818
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.3865833, upper bound: 7.3865818
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6311817, upper bound: 7.6311817
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6311824, upper bound: 7.6311813
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6312469, upper bound: 7.6312475
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.99
Output dim: 2, lower bound: -7.6312469, upper bound: 7.6312468

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2583476, upper bound: 7.2583488
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2583476, upper bound: 7.2583488
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295400, upper bound: 7.6295410
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295400, upper bound: 7.6295397
time: 2.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3548052, upper bound: 7.3548063
time: 2.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.3548052, upper bound: 7.3548063
time: 2.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295550, upper bound: 7.6295556
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6295550, upper bound: 7.6295555
time: 2.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304230, upper bound: 7.6304231
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304230, upper bound: 7.6304231
time: 2.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 65

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9876457, upper bound: 6.9876461
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9876457, upper bound: 6.9876461
time: 1.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307020, upper bound: 7.6307028
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307020, upper bound: 7.6307028
time: 3.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307211, upper bound: 7.6307220
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307212, upper bound: 7.6307214
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304170, upper bound: 7.6304170
time: 1.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6304170, upper bound: 7.6304170
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6093987, upper bound: 7.6093984
time: 2.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6093987, upper bound: 7.6093986
time: 3.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303374, upper bound: 7.6303375
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303375, upper bound: 7.6303376
time: 1.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303377, upper bound: 7.6303374
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6303377, upper bound: 7.6303375
time: 4.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6093981, upper bound: 7.6093986
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6093981, upper bound: 7.6093988
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6093983, upper bound: 7.6093982
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6093983, upper bound: 7.6093982
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305994, upper bound: 7.6306000
time: 2.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6305994, upper bound: 7.6305999
time: 2.36 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309714, upper bound: 7.6309717
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309715, upper bound: 7.6309723
time: 2.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306650, upper bound: 7.6306656
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306650, upper bound: 7.6306649
time: 2.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309785, upper bound: 7.6309790
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309786, upper bound: 7.6309790
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290867, upper bound: 7.6290878
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6290868, upper bound: 7.6290874
time: 1.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281229, upper bound: 7.6281236
time: 3.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6281229, upper bound: 7.6281240
time: 2.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309635, upper bound: 7.6309638
time: 3.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309635, upper bound: 7.6309641
time: 3.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309831, upper bound: 7.6309821
time: 2.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6309830, upper bound: 7.6309822
time: 2.39 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 6.78 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.2583476, upper bound: 7.2583488
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.2583476, upper bound: 7.2583488
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6295400, upper bound: 7.6295410
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6295400, upper bound: 7.6295397
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.3548052, upper bound: 7.3548063
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.3548052, upper bound: 7.3548063
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6295550, upper bound: 7.6295556
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6295550, upper bound: 7.6295555
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6304230, upper bound: 7.6304231
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6304230, upper bound: 7.6304231
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -6.9876457, upper bound: 6.9876461
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -6.9876457, upper bound: 6.9876461
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6307020, upper bound: 7.6307028
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6307020, upper bound: 7.6307028
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6307211, upper bound: 7.6307220
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6307212, upper bound: 7.6307214
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6304170, upper bound: 7.6304170
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6304170, upper bound: 7.6304170
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6093987, upper bound: 7.6093984
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6093987, upper bound: 7.6093986
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6303374, upper bound: 7.6303375
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6303375, upper bound: 7.6303376
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6303377, upper bound: 7.6303374
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6303377, upper bound: 7.6303375
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6093981, upper bound: 7.6093986
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6093981, upper bound: 7.6093988
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6093983, upper bound: 7.6093982
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6093983, upper bound: 7.6093982
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6305994, upper bound: 7.6306000
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6305994, upper bound: 7.6305999
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309714, upper bound: 7.6309717
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309715, upper bound: 7.6309723
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6306650, upper bound: 7.6306656
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6306650, upper bound: 7.6306649
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309785, upper bound: 7.6309790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309786, upper bound: 7.6309790
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6290867, upper bound: 7.6290878
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6290868, upper bound: 7.6290874
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6281229, upper bound: 7.6281236
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6281229, upper bound: 7.6281240
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309635, upper bound: 7.6309638
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309635, upper bound: 7.6309641
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309831, upper bound: 7.6309821
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 6.78
Output dim: 2, lower bound: -7.6309830, upper bound: 7.6309822
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6307159, upper bound: 7.6307160
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6307159, upper bound: 7.6307159
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6309633, upper bound: 7.6309640
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6309633, upper bound: 7.6309640
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6305847, upper bound: 7.6305849
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6305848, upper bound: 7.6305845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6311817, upper bound: 7.6311817
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6311824, upper bound: 7.6311813
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6312469, upper bound: 7.6312475
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 2, lower bound: -7.6312469, upper bound: 7.6312468
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=9.388381958007812
rel_dist={2: [-7.633595790761577, 7.633596173769259]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6331834, upper bound: 7.6331834
time: 11.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6331834, upper bound: 7.6331834
time: 2.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.75
Output dim: 2, lower bound: -7.6331834, upper bound: 7.6331834
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.75
Output dim: 2, lower bound: -7.6331834, upper bound: 7.6331834

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6331390, upper bound: 7.6331388
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6331390, upper bound: 7.6331392
time: 3.55 seconds

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
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323166, upper bound: 7.6323165
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323166, upper bound: 7.6323171
time: 9.41 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -7.6331390, upper bound: 7.6331388
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -7.6331390, upper bound: 7.6331392
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -7.6323166, upper bound: 7.6323165
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.09
Output dim: 2, lower bound: -7.6323166, upper bound: 7.6323171

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
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 96

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311806, upper bound: 7.6311805
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311806, upper bound: 7.6311805
time: 4.20 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313461
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313461
time: 3.72 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320280
time: 4.07 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323148, upper bound: 7.6323149
time: 3.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323148, upper bound: 7.6323153
time: 3.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 9.53 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6311806, upper bound: 7.6311805
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6311806, upper bound: 7.6311805
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313461
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313461
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320280
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6323148, upper bound: 7.6323149
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 9.53
Output dim: 2, lower bound: -7.6323148, upper bound: 7.6323153

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4091336, upper bound: 7.4091340
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.4091336, upper bound: 7.4091340
time: 2.43 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275776
time: 2.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275774
time: 2.60 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311725, upper bound: 7.6311732
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311725, upper bound: 7.6311731
time: 2.87 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313461
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313454
time: 2.90 seconds

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320280
time: 3.30 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320285
time: 6.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
time: 4.75 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323148, upper bound: 7.6323149
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323149, upper bound: 7.6323154
time: 2.62 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320271
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320271
time: 6.94 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.30 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.4091336, upper bound: 7.4091340
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.4091336, upper bound: 7.4091340
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275776
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275774
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6311725, upper bound: 7.6311732
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6311725, upper bound: 7.6311731
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313461
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6313453, upper bound: 7.6313454
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320280
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320285
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6323148, upper bound: 7.6323149
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6323149, upper bound: 7.6323154
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320271
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.30
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320271

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6153428, upper bound: 7.6153428
time: 3.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6153428, upper bound: 7.6153428
time: 3.14 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275774
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275774, upper bound: 7.6275777
time: 3.69 seconds

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
time: 4.25 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311731, upper bound: 7.6311732
time: 2.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311731, upper bound: 7.6311732
time: 4.08 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 65

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9720606, upper bound: 6.9720607
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -6.9720606, upper bound: 6.9720607
time: 1.73 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306101, upper bound: 7.6306101
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306101, upper bound: 7.6306101
time: 2.35 seconds

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
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5612442, upper bound: 7.5612433
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5612442, upper bound: 7.5612433
time: 2.47 seconds

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
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 247

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315335, upper bound: 7.6315334
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315335, upper bound: 7.6315335
time: 4.31 seconds

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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5761998, upper bound: 7.5761974
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5761998, upper bound: 7.5761976
time: 3.89 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320286
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
time: 4.00 seconds

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287941, upper bound: 7.6287934
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6287941, upper bound: 7.6287934
time: 3.27 seconds

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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323146
time: 2.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323139
time: 4.33 seconds

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
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320271
time: 2.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320272
time: 2.87 seconds

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
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 173

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319648, upper bound: 7.6319643
time: 9.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319648, upper bound: 7.6319642
time: 5.62 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.28 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6153428, upper bound: 7.6153428
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6153428, upper bound: 7.6153428
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275774
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6275774, upper bound: 7.6275777
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6311731, upper bound: 7.6311732
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6311731, upper bound: 7.6311732
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -6.9720606, upper bound: 6.9720607
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -6.9720606, upper bound: 6.9720607
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6306101, upper bound: 7.6306101
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6306101, upper bound: 7.6306101
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.5612442, upper bound: 7.5612433
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.5612442, upper bound: 7.5612433
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6315335, upper bound: 7.6315334
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6315335, upper bound: 7.6315335
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.5761998, upper bound: 7.5761974
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.5761998, upper bound: 7.5761976
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320286
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6320279, upper bound: 7.6320279
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6287941, upper bound: 7.6287934
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6287941, upper bound: 7.6287934
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323146
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323139
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320271
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6320265, upper bound: 7.6320272
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6319648, upper bound: 7.6319643
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.28
Output dim: 2, lower bound: -7.6319648, upper bound: 7.6319642

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275776, upper bound: 7.6275774
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275773
time: 2.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6167993, upper bound: 7.6167992
time: 3.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6167993, upper bound: 7.6167992
time: 2.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311633, upper bound: 7.6311632
time: 3.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311633, upper bound: 7.6311633
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2973449, upper bound: 7.2973450
time: 3.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2973449, upper bound: 7.2973450
time: 3.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311625, upper bound: 7.6311627
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311625, upper bound: 7.6311627
time: 3.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307959
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307951
time: 5.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
time: 5.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306100
time: 2.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
time: 3.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315336, upper bound: 7.6315329
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315336, upper bound: 7.6315334
time: 5.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315334, upper bound: 7.6315328
time: 6.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6315335, upper bound: 7.6315334
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320284, upper bound: 7.6320286
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320279
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320285
time: 2.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320285
time: 6.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5756286, upper bound: 7.5756295
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5756291, upper bound: 7.5756291
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285800, upper bound: 7.6285800
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6285802, upper bound: 7.6285798
time: 3.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6122141, upper bound: 7.6122126
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6122141, upper bound: 7.6122130
time: 4.44 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323145
time: 3.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323145
time: 2.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320272, upper bound: 7.6320265
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320272, upper bound: 7.6320271
time: 2.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320271, upper bound: 7.6320265
time: 5.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6320271, upper bound: 7.6320272
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319647, upper bound: 7.6319648
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6319648, upper bound: 7.6319647
time: 5.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 211

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314558, upper bound: 7.6314561
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6314561, upper bound: 7.6314558
time: 3.70 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 9.44 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6275776, upper bound: 7.6275774
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6275777, upper bound: 7.6275773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6167993, upper bound: 7.6167992
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6167993, upper bound: 7.6167992
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6311633, upper bound: 7.6311632
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6311633, upper bound: 7.6311633
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.2973449, upper bound: 7.2973450
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.2973449, upper bound: 7.2973450
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6311625, upper bound: 7.6311627
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6311625, upper bound: 7.6311627
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307959
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307951
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306100
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6315336, upper bound: 7.6315329
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6315336, upper bound: 7.6315334
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6315334, upper bound: 7.6315328
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6315335, upper bound: 7.6315334
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320284, upper bound: 7.6320286
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320279
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320285
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320285
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.5756286, upper bound: 7.5756295
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.5756291, upper bound: 7.5756291
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6285800, upper bound: 7.6285800
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6285802, upper bound: 7.6285798
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6122141, upper bound: 7.6122126
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6122141, upper bound: 7.6122130
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323145
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323145
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320272, upper bound: 7.6320265
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320272, upper bound: 7.6320271
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320271, upper bound: 7.6320265
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6320271, upper bound: 7.6320272
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6319647, upper bound: 7.6319648
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6319648, upper bound: 7.6319647
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6314558, upper bound: 7.6314561
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 9.44
Output dim: 2, lower bound: -7.6314561, upper bound: 7.6314558

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 174

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6114231, upper bound: 7.6114228
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.6114231, upper bound: 7.6114226
time: 2.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2804346, upper bound: 7.2804347
time: 2.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2804346, upper bound: 7.2804347
time: 2.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311625, upper bound: 7.6311633
time: 8.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
time: 6.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2973450, upper bound: 7.2973449
time: 2.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.2973450, upper bound: 7.2973454
time: 5.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311632
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
time: 7.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5165553, upper bound: 7.5165562
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -7.5165553, upper bound: 7.5165562
time: 3.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 58
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 58

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307959
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307958
time: 3.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 58

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307848, upper bound: 7.6307855
time: 18.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6307848, upper bound: 7.6307854
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6297997, upper bound: 7.6297996
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6297997, upper bound: 7.6297996
time: 2.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 211
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
time: 2.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306100
time: 5.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 96
type: RSZ, layer: 1, pos: 65
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306100
time: 4.17 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 10.42 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6114231, upper bound: 7.6114228
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6114231, upper bound: 7.6114226
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.2804346, upper bound: 7.2804347
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.2804346, upper bound: 7.2804347
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6311625, upper bound: 7.6311633
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.2973450, upper bound: 7.2973449
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.2973450, upper bound: 7.2973454
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311632
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6311632, upper bound: 7.6311626
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.5165553, upper bound: 7.5165562
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.5165553, upper bound: 7.5165562
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307959
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6307958, upper bound: 7.6307958
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6307848, upper bound: 7.6307855
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6307848, upper bound: 7.6307854
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6297997, upper bound: 7.6297996
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6297997, upper bound: 7.6297996
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306100
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 10.42
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306100
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6306100, upper bound: 7.6306101
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6315336, upper bound: 7.6315329
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6315336, upper bound: 7.6315334
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6315334, upper bound: 7.6315328
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6315335, upper bound: 7.6315334
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320284, upper bound: 7.6320286
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320279
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320285
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320285, upper bound: 7.6320285
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6285800, upper bound: 7.6285800
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6285802, upper bound: 7.6285798
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323145
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6323146, upper bound: 7.6323145
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320272, upper bound: 7.6320265
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320272, upper bound: 7.6320271
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320271, upper bound: 7.6320265
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6320271, upper bound: 7.6320272
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6319647, upper bound: 7.6319648
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6319648, upper bound: 7.6319647
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6314558, upper bound: 7.6314561
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 10.42
Output dim: 2, lower bound: -7.6314561, upper bound: 7.6314558
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=9.388381958007812
rel_dist={2: [-7.63358222082411, 7.633582568333054]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1808.26 seconds
