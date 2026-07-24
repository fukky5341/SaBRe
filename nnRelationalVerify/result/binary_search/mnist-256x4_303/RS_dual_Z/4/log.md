## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 10.8418399842
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673)
1: (-5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646)
2: (-7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834)
3: (-7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134)
4: (-8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527)
5: (-7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887)
6: (-6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578)
7: (-7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368)
8: (-9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864)
9: (-6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577)

## BASE Result
execution time: IAR + LP analysis = 1.19 + 4.11 = 5.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -10.8418977, upper bound: 10.8418970


# Binary Search by BASE starts (time budget: 2694.70 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=13.22586727142334
rel_dist={0: [-10.841897398404505, 10.841898463369382]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=13.22586727142334
rel_dist={0: [-10.84189901486047, 10.841897299947618]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=13.22586727142334
rel_dist={0: [-10.841896423745458, 10.841896847270611]}

## Binary Search Result
Binary search time: 19.05 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 2675.65 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418512, upper bound: 10.8418501
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418512, upper bound: 10.8418503
time: 3.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.17 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.17
Output dim: 0, lower bound: -10.8418512, upper bound: 10.8418501
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.17
Output dim: 0, lower bound: -10.8418512, upper bound: 10.8418503

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418507
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418510
time: 2.58 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418509
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418505
time: 2.39 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.73 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.73
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418507
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.73
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418510
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 5.73
Output dim: 0, lower bound: -10.8418500, upper bound: 10.8418509
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 5.73
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418505

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418495
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418472
time: 2.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418487
time: 2.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418496
time: 2.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418488
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418488
time: 1.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418470
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418496, upper bound: 10.8418496
time: 2.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.09 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418495
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418472
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418487
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418496
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418488
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418488
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418470
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.09
Output dim: 0, lower bound: -10.8418496, upper bound: 10.8418496

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418172, upper bound: 10.8418175
time: 2.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418183
time: 2.52 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
time: 2.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
time: 2.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418184
time: 2.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418184
time: 2.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418184
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418185
time: 2.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418182
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418177
time: 2.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
time: 3.33 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418176
time: 2.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418176
time: 2.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418184
time: 2.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418185
time: 2.96 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 6.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418172, upper bound: 10.8418175
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418183
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418184
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418184
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418184
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418185
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418182
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418168
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418176
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418176
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418184
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 6.79
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418185
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=13.22586727142334
rel_dist={0: [-10.841897398404505, 10.841898463369382]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418501
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418500
time: 2.81 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.76 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.76
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418501
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.76
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418500

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418504
time: 2.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418508
time: 2.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418509
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418508
time: 2.69 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.06 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418504
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418508
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -10.8418503, upper bound: 10.8418509
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.06
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418508

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418498, upper bound: 10.8418497
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418491
time: 2.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418495, upper bound: 10.8418491
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418496, upper bound: 10.8418484
time: 2.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418474
time: 2.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418493
time: 2.32 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418495, upper bound: 10.8418491
time: 1.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418496, upper bound: 10.8418484
time: 2.51 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.65 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418498, upper bound: 10.8418497
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418491
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418495, upper bound: 10.8418491
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418496, upper bound: 10.8418484
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418474
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418493
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418495, upper bound: 10.8418491
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.65
Output dim: 0, lower bound: -10.8418496, upper bound: 10.8418484

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418186
time: 3.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418184
time: 1.70 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
time: 2.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
time: 2.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418179
time: 2.09 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
time: 1.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418186
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418184
time: 1.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418177
time: 2.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
time: 2.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418179
time: 2.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418166
time: 2.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
time: 2.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
time: 2.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418186
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418184
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418179
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418186
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418184
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418177
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418183
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418179
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418180, upper bound: 10.8418166
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.61
Output dim: 0, lower bound: -10.8418179, upper bound: 10.8418183
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=13.22586727142334
rel_dist={0: [-10.841897768243273, 10.841897448887977]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418505
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418506
time: 1.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.07 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.07
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418505
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.07
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418506

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418506, upper bound: 10.8418503
time: 1.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418501, upper bound: 10.8418492
time: 1.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418506, upper bound: 10.8418511
time: 2.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418501, upper bound: 10.8418492
time: 2.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.37 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 0, lower bound: -10.8418506, upper bound: 10.8418503
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 0, lower bound: -10.8418501, upper bound: 10.8418492
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 0, lower bound: -10.8418506, upper bound: 10.8418511
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.37
Output dim: 0, lower bound: -10.8418501, upper bound: 10.8418492

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418491
time: 2.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418496
time: 2.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418484
time: 2.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418490
time: 2.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418493
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418494
time: 2.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418484
time: 2.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418482
time: 2.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 6.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418491
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418496
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418484
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418490
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418493
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418494
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418491, upper bound: 10.8418484
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 6.88
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418482

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418185
time: 1.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418177
time: 1.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
time: 1.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
time: 1.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418178
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418171
time: 1.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418170
time: 2.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
time: 1.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418185
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418177
time: 1.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
time: 1.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418177
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418181
time: 2.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418170
time: 2.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
time: 1.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.31 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418185
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418177
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418178
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418171
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418170
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418185
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418167, upper bound: 10.8418177
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418185, upper bound: 10.8418178
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418177
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418175, upper bound: 10.8418181
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418170
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.31
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=13.22586727142334
rel_dist={0: [-10.84189845279027, 10.841897140740066]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418509
time: 1.50 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418509
time: 1.52 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.16 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.16
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418509
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.16
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418509

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418497
time: 1.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418508
time: 2.08 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418507
time: 1.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418508
time: 2.07 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.86 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418497
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418508
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418507
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.86
Output dim: 0, lower bound: -10.8418509, upper bound: 10.8418508

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418482
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418490
time: 1.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418490, upper bound: 10.8418490
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418473
time: 2.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418482
time: 1.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418493
time: 1.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418490, upper bound: 10.8418490
time: 1.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418492
time: 2.15 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.35 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418482
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418490
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418490, upper bound: 10.8418490
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418473
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418497, upper bound: 10.8418482
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418493, upper bound: 10.8418493
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418490, upper bound: 10.8418490
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.35
Output dim: 0, lower bound: -10.8418492, upper bound: 10.8418492

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
time: 1.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
time: 1.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
time: 1.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
time: 1.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418181
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
time: 1.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
time: 1.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
time: 1.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
time: 1.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418181
time: 2.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.7459450, 4.4799228, -8.7459450, 4.4799228, -13.2258663, 13.2258673
1: -5.7760954, 5.5418692, -5.7760954, 5.5418692, -11.3179646, 11.3179646
2: -7.5248165, 5.6160679, -7.5248165, 5.6160679, -13.1408844, 13.1408834
3: -7.9764252, 4.9407926, -7.9764252, 4.9407926, -12.9172134, 12.9172134
4: -8.6739941, 6.9140596, -8.6739941, 6.9140596, -15.5880537, 15.5880527
5: -7.1571879, 5.0683022, -7.1571879, 5.0683022, -12.2254887, 12.2254887
6: -6.6823978, 6.7787600, -6.6823978, 6.7787600, -13.4611568, 13.4611578
7: -7.3757725, 6.7552648, -7.3757725, 6.7552648, -14.1310368, 14.1310368
8: -9.3987417, 5.7281442, -9.3987417, 5.7281442, -15.1268864, 15.1268864
9: -6.5382051, 6.7157540, -6.5382051, 6.7157540, -13.2539577, 13.2539577

Time for backsubstitution: 1.19 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
time: 1.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
time: 1.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 4.80 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418181
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418180
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418177, upper bound: 10.8418185
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418184
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418184, upper bound: 10.8418181
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 0, lower bound: -10.8418174, upper bound: 10.8418176
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=13.22586727142334
rel_dist={0: [-10.84189768916142, 10.841896961046885]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 365.44 seconds
