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
execution time: IAR + LP analysis = 1.22 + 4.51 = 5.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -10.8418977, upper bound: 10.8418970


# Binary Search by BASE starts (time budget: 2694.27 seconds, max iter: 100)

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
Binary search time: 18.80 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 2675.46 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418966, upper bound: 10.8418984
time: 2.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418963, upper bound: 10.8418964
time: 2.27 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 4.98 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 4.98
Output dim: 0, lower bound: -10.8418966, upper bound: 10.8418984
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 4.98
Output dim: 0, lower bound: -10.8418963, upper bound: 10.8418964

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417736, upper bound: 10.8417726
time: 1.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417736, upper bound: 10.8417726
time: 1.77 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418195, upper bound: 10.8418198
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418195, upper bound: 10.8418198
time: 2.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 5.69 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 5.69
Output dim: 0, lower bound: -10.8417736, upper bound: 10.8417726
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 5.69
Output dim: 0, lower bound: -10.8417736, upper bound: 10.8417726
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 5.69
Output dim: 0, lower bound: -10.8418195, upper bound: 10.8418198
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 5.69
Output dim: 0, lower bound: -10.8418195, upper bound: 10.8418198
Binary search (step 0): status=Status.VERIFIED, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=13.22586727142334
rel_dist={0: [-10.841897398404505, 10.841898463369382]}

## Binary search (step 1) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418616, upper bound: 10.8418608
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418610, upper bound: 10.8418622
time: 3.83 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.20 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.20
Output dim: 0, lower bound: -10.8418616, upper bound: 10.8418608
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.20
Output dim: 0, lower bound: -10.8418610, upper bound: 10.8418622

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418309, upper bound: 10.8418303
time: 3.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418299, upper bound: 10.8418296
time: 2.66 seconds

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 109

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418612, upper bound: 10.8418620
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418621, upper bound: 10.8418625
time: 2.43 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.30 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 6.30
Output dim: 0, lower bound: -10.8418309, upper bound: 10.8418303
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 6.30
Output dim: 0, lower bound: -10.8418299, upper bound: 10.8418296
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.30
Output dim: 0, lower bound: -10.8418612, upper bound: 10.8418620
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.30
Output dim: 0, lower bound: -10.8418621, upper bound: 10.8418625

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418611, upper bound: 10.8418611
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418611, upper bound: 10.8418617
time: 2.73 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418360
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418360
time: 2.45 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.98 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 0, lower bound: -10.8418611, upper bound: 10.8418611
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.98
Output dim: 0, lower bound: -10.8418611, upper bound: 10.8418617
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 5.98
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418360
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 5.98
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418360

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418621, upper bound: 10.8418623
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418624, upper bound: 10.8418617
time: 4.64 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418171, upper bound: 10.8418162
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418171, upper bound: 10.8418161
time: 2.35 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 5.81 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 5.81
Output dim: 0, lower bound: -10.8418621, upper bound: 10.8418623
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 5.81
Output dim: 0, lower bound: -10.8418624, upper bound: 10.8418617
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 5.81
Output dim: 0, lower bound: -10.8418171, upper bound: 10.8418162
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 5.81
Output dim: 0, lower bound: -10.8418171, upper bound: 10.8418161

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417252, upper bound: 10.8417246
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417252, upper bound: 10.8417246
time: 2.72 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418567, upper bound: 10.8418578
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418575, upper bound: 10.8418573
time: 2.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 6.21 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 6.21
Output dim: 0, lower bound: -10.8417252, upper bound: 10.8417246
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 6.21
Output dim: 0, lower bound: -10.8417252, upper bound: 10.8417246
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 6.21
Output dim: 0, lower bound: -10.8418567, upper bound: 10.8418578
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 6.21
Output dim: 0, lower bound: -10.8418575, upper bound: 10.8418573

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418526, upper bound: 10.8418526
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418523, upper bound: 10.8418528
time: 2.46 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417371, upper bound: 10.8417376
time: 2.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417371, upper bound: 10.8417365
time: 2.16 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.63 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 0, lower bound: -10.8418526, upper bound: 10.8418526
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.63
Output dim: 0, lower bound: -10.8418523, upper bound: 10.8418528
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 0, lower bound: -10.8417371, upper bound: 10.8417376
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.63
Output dim: 0, lower bound: -10.8417371, upper bound: 10.8417365

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417324, upper bound: 10.8417320
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417324, upper bound: 10.8417322
time: 2.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418430, upper bound: 10.8418424
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418430, upper bound: 10.8418411
time: 2.46 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 8.38 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 8.38
Output dim: 0, lower bound: -10.8417324, upper bound: 10.8417320
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 8.38
Output dim: 0, lower bound: -10.8417324, upper bound: 10.8417322
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 8.38
Output dim: 0, lower bound: -10.8418430, upper bound: 10.8418424
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 8.38
Output dim: 0, lower bound: -10.8418430, upper bound: 10.8418411

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 168

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418350
time: 2.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418347
time: 3.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417975, upper bound: 10.8417974
time: 2.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417975, upper bound: 10.8417974
time: 2.79 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 8.75 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 8.75
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418350
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 8.75
Output dim: 0, lower bound: -10.8418351, upper bound: 10.8418347
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 8.75
Output dim: 0, lower bound: -10.8417975, upper bound: 10.8417974
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 8.75
Output dim: 0, lower bound: -10.8417975, upper bound: 10.8417974
Binary search (step 1): status=Status.VERIFIED, k_low=7, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=13.22586727142334
rel_dist={0: [-10.841897768243273, 10.841897448887977]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418918, upper bound: 10.8418922
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418917, upper bound: 10.8418922
time: 1.97 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.91 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.91
Output dim: 0, lower bound: -10.8418918, upper bound: 10.8418922
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.91
Output dim: 0, lower bound: -10.8418917, upper bound: 10.8418922

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418436
time: 2.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418437
time: 2.00 seconds

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

Time for backsubstitution: 1.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 53

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418926, upper bound: 10.8418911
time: 2.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418931, upper bound: 10.8418923
time: 3.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.12
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418436
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.12
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418437
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.12
Output dim: 0, lower bound: -10.8418926, upper bound: 10.8418911
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.12
Output dim: 0, lower bound: -10.8418931, upper bound: 10.8418923

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418437
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418436
time: 2.22 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418397
time: 2.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418369
time: 2.05 seconds

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
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418665, upper bound: 10.8418651
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418659, upper bound: 10.8418639
time: 2.50 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418433, upper bound: 10.8418418
time: 1.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418433, upper bound: 10.8418436
time: 2.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 5.16 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418437
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418436, upper bound: 10.8418436
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418397
RS_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418392, upper bound: 10.8418369
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418665, upper bound: 10.8418651
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418659, upper bound: 10.8418639
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418433, upper bound: 10.8418418
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 5.16
Output dim: 0, lower bound: -10.8418433, upper bound: 10.8418436

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

Time for backsubstitution: 1.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418412, upper bound: 10.8418418
time: 1.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418413, upper bound: 10.8418396
time: 1.82 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 135

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418435
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418430, upper bound: 10.8418435
time: 2.05 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418660
time: 2.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418650
time: 1.79 seconds

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

Time for backsubstitution: 1.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 70

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418247, upper bound: 10.8418246
time: 2.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418247, upper bound: 10.8418254
time: 2.20 seconds

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

Time for backsubstitution: 1.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 165

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417720, upper bound: 10.8417725
time: 1.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417720, upper bound: 10.8417716
time: 1.61 seconds

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

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418056, upper bound: 10.8418055
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418056, upper bound: 10.8418046
time: 2.07 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418412, upper bound: 10.8418418
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418413, upper bound: 10.8418396
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418426, upper bound: 10.8418435
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418430, upper bound: 10.8418435
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418660
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418656, upper bound: 10.8418650
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418247, upper bound: 10.8418246
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418247, upper bound: 10.8418254
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8417720, upper bound: 10.8417725
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8417720, upper bound: 10.8417716
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418056, upper bound: 10.8418055
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 7.63
Output dim: 0, lower bound: -10.8418056, upper bound: 10.8418046

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418416
time: 1.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418421
time: 2.51 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 255

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418415
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418418, upper bound: 10.8418406
time: 2.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8418341
time: 2.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8418332
time: 2.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 83

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8418350
time: 2.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418340, upper bound: 10.8418349
time: 2.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417458, upper bound: 10.8417459
time: 1.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417458, upper bound: 10.8417459
time: 1.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418397
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418397
time: 2.42 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 5.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418416
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418421
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418415
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418418, upper bound: 10.8418406
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8418341
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8418332
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418346, upper bound: 10.8418350
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418340, upper bound: 10.8418349
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8417458, upper bound: 10.8417459
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8417458, upper bound: 10.8417459
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418397
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 5.83
Output dim: 0, lower bound: -10.8418383, upper bound: 10.8418397

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418102, upper bound: 10.8418094
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418102, upper bound: 10.8418079
time: 3.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418043, upper bound: 10.8418038
time: 2.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418043, upper bound: 10.8418038
time: 1.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 249

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418048, upper bound: 10.8418053
time: 2.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418048, upper bound: 10.8418053
time: 2.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418418, upper bound: 10.8418416
time: 2.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418415, upper bound: 10.8418414
time: 2.24 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418102, upper bound: 10.8418094
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418102, upper bound: 10.8418079
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418043, upper bound: 10.8418038
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418043, upper bound: 10.8418038
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418048, upper bound: 10.8418053
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418048, upper bound: 10.8418053
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418418, upper bound: 10.8418416
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.76
Output dim: 0, lower bound: -10.8418415, upper bound: 10.8418414

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417569, upper bound: 10.8417562
time: 2.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417569, upper bound: 10.8417568
time: 2.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418415, upper bound: 10.8418412
time: 2.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418407
time: 2.33 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 5.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 5.63
Output dim: 0, lower bound: -10.8417569, upper bound: 10.8417562
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 5.63
Output dim: 0, lower bound: -10.8417569, upper bound: 10.8417568
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 5.63
Output dim: 0, lower bound: -10.8418415, upper bound: 10.8418412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 5.63
Output dim: 0, lower bound: -10.8418414, upper bound: 10.8418407

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417934, upper bound: 10.8417932
time: 3.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417934, upper bound: 10.8417932
time: 2.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 90

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418140, upper bound: 10.8418144
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418140, upper bound: 10.8418144
time: 2.14 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 7.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -10.8417934, upper bound: 10.8417932
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -10.8417934, upper bound: 10.8417932
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -10.8418140, upper bound: 10.8418144
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 7.90
Output dim: 0, lower bound: -10.8418140, upper bound: 10.8418144
Binary search (step 2): status=Status.VERIFIED, k_low=10, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=13.22586727142334
rel_dist={0: [-10.84189845279027, 10.841897140740066]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 210
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 210

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418976, upper bound: 10.8418985
time: 1.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418976, upper bound: 10.8418973
time: 1.84 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 3.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 3.58
Output dim: 0, lower bound: -10.8418976, upper bound: 10.8418985
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 3.58
Output dim: 0, lower bound: -10.8418976, upper bound: 10.8418973

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
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 69

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417725, upper bound: 10.8417705
time: 2.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417725, upper bound: 10.8417705
time: 2.32 seconds

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

Time for backsubstitution: 1.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418503
time: 1.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418506
time: 1.84 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.85 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.85
Output dim: 0, lower bound: -10.8417725, upper bound: 10.8417705
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.85
Output dim: 0, lower bound: -10.8417725, upper bound: 10.8417705
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.85
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418503
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.85
Output dim: 0, lower bound: -10.8418508, upper bound: 10.8418506

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

Time for backsubstitution: 1.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417850, upper bound: 10.8417852
time: 1.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8417852, upper bound: 10.8417856
time: 2.03 seconds

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
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 165
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 235
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 70
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 90
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 255
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 135
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 53

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 242

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418007, upper bound: 10.8418019
time: 2.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -10.8418007, upper bound: 10.8418019
time: 2.04 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.39 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 11.39
Output dim: 0, lower bound: -10.8417850, upper bound: 10.8417852
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 11.39
Output dim: 0, lower bound: -10.8417852, upper bound: 10.8417856
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 11.39
Output dim: 0, lower bound: -10.8418007, upper bound: 10.8418019
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 11.39
Output dim: 0, lower bound: -10.8418007, upper bound: 10.8418019
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=13.22586727142334
rel_dist={0: [-10.84189768916142, 10.841896961046885]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 331.32 seconds
