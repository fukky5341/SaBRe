## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_3.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 19342.684212278527


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125)
1: (-12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062)
2: (-18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625)
3: (-6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750)
4: (-20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625)

## BASE Result
execution time: IAR + LP analysis = 1.58 + 2.36 = 3.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -19343.1395978, upper bound: 19343.1395978


# Binary Search by BASE starts (time budget: 1196.05 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=24289.171875
rel_dist={3: [-19343.13957058345, 19343.13957058345]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=24289.171875
rel_dist={3: [-19343.120444796758, 19343.120444796754]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=24289.171875
rel_dist={3: [-19343.089144719655, 19343.089144719655]}

## Binary search (step 3) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=24289.171875
rel_dist={3: [-19343.02999653799, 19343.029996537996]}

## Binary search (step 4) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=24289.171875
rel_dist={3: [-19342.98255923941, 19342.98255923942]}

## Binary search (step 5) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=24289.171875
rel_dist={3: [-19342.956835158915, 19342.95683515891]}

## Binary search (step 6) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=24289.171875
rel_dist={3: [-19342.94283207538, 19342.94283207538]}

## Binary search (step 7) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=24289.171875
rel_dist={3: [-19342.93540470574, 19342.935404705742]}

## Binary search (step 8) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=24289.171875
rel_dist={3: [-19342.931612648863, 19342.931612648863]}

## Binary search (step 9) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=24289.171875
rel_dist={3: [-19342.929656883058, 19342.92965688306]}

## Binary search (step 10) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=24289.171875
rel_dist={3: [-19342.92866784536, 19342.928667845365]}

## Binary search (step 11) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=24289.171875
rel_dist={3: [-19342.92817332682, 19342.92817332683]}

## Binary search (step 12) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=24289.171875
rel_dist={3: [-19342.92792606815, 19342.92792606815]}

## Binary search (step 13) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=24289.171875
rel_dist={3: [-19342.92780244, 19342.927802440005]}

## Binary search (step 14) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=24289.171875
rel_dist={3: [-19342.927740628278, 19342.927740628278]}

## Binary search (step 15) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=24289.171875
rel_dist={3: [-19342.927709727028, 19342.927709727024]}

## Binary search (step 16) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=24289.171875
rel_dist={3: [-19342.927694285212, 19342.927694296057]}

## Binary search (step 17) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=24289.171875
rel_dist={3: [-19342.927686580566, 19342.927686589654]}

## Binary search (step 18) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 18): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=24289.171875
rel_dist={3: [-19342.927682844893, 19342.927682850517]}

## Binary search (step 19) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 19): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=24289.171875
rel_dist={3: [-19342.92768119177, 19342.927681716894]}

## Binary Search Result
Binary search time: 79.97 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1116.09 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9249007, upper bound: 19342.9249007
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9249007, upper bound: 19342.9249007
time: 0.87 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.69 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 3, lower bound: -19342.9249007, upper bound: 19342.9249007
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 3, lower bound: -19342.9249007, upper bound: 19342.9249007

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9036773, upper bound: 19342.9085367
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9084750, upper bound: 19342.9036773
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8576444, upper bound: 19342.8553201
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8554171, upper bound: 19342.8576431
time: 0.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -19342.9036773, upper bound: 19342.9085367
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -19342.9084750, upper bound: 19342.9036773
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -19342.8576444, upper bound: 19342.8553201
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.15
Output dim: 3, lower bound: -19342.8554171, upper bound: 19342.8576431

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9011767, upper bound: 19342.9062631
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9014424, upper bound: 19342.9027133
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8646821, upper bound: 19342.8591387
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8646821, upper bound: 19342.8591387
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7775167
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7775167
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7871379, upper bound: 19342.7930346
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7883669, upper bound: 19342.7920589
time: 0.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.31 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.9011767, upper bound: 19342.9062631
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.9014424, upper bound: 19342.9027133
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.8646821, upper bound: 19342.8591387
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.8646821, upper bound: 19342.8591387
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7775167
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7775167
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.7871379, upper bound: 19342.7930346
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.31
Output dim: 3, lower bound: -19342.7883669, upper bound: 19342.7920589

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8888799, upper bound: 19342.8918152
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8888799, upper bound: 19342.8918152
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8924358, upper bound: 19342.8832762
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8924358, upper bound: 19342.8833119
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8584618, upper bound: 19342.8560671
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8615701, upper bound: 19342.8561825
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7710548, upper bound: 19342.7692479
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7710548, upper bound: 19342.7692479
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7763336
time: 1.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7693291, upper bound: 19342.7775167
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7250046, upper bound: 19342.7251714
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7251731, upper bound: 19342.7249553
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7842599, upper bound: 19342.7847021
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7842604, upper bound: 19342.7855644
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3923355, upper bound: 19342.3903109
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3923355, upper bound: 19342.3903109
time: 0.86 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.23 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.8888799, upper bound: 19342.8918152
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.8888799, upper bound: 19342.8918152
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.8924358, upper bound: 19342.8832762
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.8924358, upper bound: 19342.8833119
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.8584618, upper bound: 19342.8560671
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.8615701, upper bound: 19342.8561825
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7710548, upper bound: 19342.7692479
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7710548, upper bound: 19342.7692479
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7763336
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7693291, upper bound: 19342.7775167
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7250046, upper bound: 19342.7251714
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7251731, upper bound: 19342.7249553
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7842599, upper bound: 19342.7847021
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.7842604, upper bound: 19342.7855644
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.3923355, upper bound: 19342.3903109
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 3, lower bound: -19342.3923355, upper bound: 19342.3903109

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8848936, upper bound: 19342.8908995
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8881502, upper bound: 19342.8846297
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8763824, upper bound: 19342.8876617
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8761115, upper bound: 19342.8876617
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8722896, upper bound: 19342.8636767
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8731712, upper bound: 19342.8639491
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8451236
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8451236
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8167997, upper bound: 19342.8168585
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8171401, upper bound: 19342.8166329
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6412789, upper bound: 19342.6404199
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6412789, upper bound: 19342.6404199
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6692345, upper bound: 19342.6687226
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6692345, upper bound: 19342.6687226
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7563984, upper bound: 19342.7544128
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7545239, upper bound: 19342.7545898
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7707999, upper bound: 19342.7762319
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7763309
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7482020, upper bound: 19342.7559795
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7481859, upper bound: 19342.7565800
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7174113, upper bound: 19342.7176308
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7174152, upper bound: 19342.7175083
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7174039, upper bound: 19342.7170370
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7175086, upper bound: 19342.7174693
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7842060, upper bound: 19342.7844257
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7842060, upper bound: 19342.7844257
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7842604, upper bound: 19342.7853728
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7840675, upper bound: 19342.7855644
time: 0.80 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8848936, upper bound: 19342.8908995
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8881502, upper bound: 19342.8846297
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8763824, upper bound: 19342.8876617
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8761115, upper bound: 19342.8876617
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8722896, upper bound: 19342.8636767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8731712, upper bound: 19342.8639491
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8451236
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8451236
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8167997, upper bound: 19342.8168585
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.8171401, upper bound: 19342.8166329
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.6412789, upper bound: 19342.6404199
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.6412789, upper bound: 19342.6404199
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.6692345, upper bound: 19342.6687226
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.6692345, upper bound: 19342.6687226
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7563984, upper bound: 19342.7544128
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7545239, upper bound: 19342.7545898
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7707999, upper bound: 19342.7762319
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7765062, upper bound: 19342.7763309
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7482020, upper bound: 19342.7559795
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7481859, upper bound: 19342.7565800
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7174113, upper bound: 19342.7176308
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7174152, upper bound: 19342.7175083
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7174039, upper bound: 19342.7170370
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7175086, upper bound: 19342.7174693
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7842060, upper bound: 19342.7844257
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7842060, upper bound: 19342.7844257
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7842604, upper bound: 19342.7853728
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.11
Output dim: 3, lower bound: -19342.7840675, upper bound: 19342.7855644

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8848054, upper bound: 19342.8797762
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8792847, upper bound: 19342.8908113
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8881502, upper bound: 19342.8796721
time: 1.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8881488, upper bound: 19342.8817691
time: 0.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8582270, upper bound: 19342.8693423
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8690178
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7839750, upper bound: 19342.7916979
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7851263, upper bound: 19342.7916979
time: 1.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8714615, upper bound: 19342.8561331
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8714615, upper bound: 19342.8561331
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8636767, upper bound: 19342.8636767
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8731712, upper bound: 19342.8639491
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8481322, upper bound: 19342.8451236
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8399633, upper bound: 19342.8396467
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8396753, upper bound: 19342.8450505
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8451236
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7771784, upper bound: 19342.7761564
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7759832, upper bound: 19342.7759832
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7983496, upper bound: 19342.7981048
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7983496, upper bound: 19342.7981048
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5378771, upper bound: 19342.5358883
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5378771, upper bound: 19342.5358883
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7517201, upper bound: 19342.7516591
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7516591, upper bound: 19342.7519041
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7335290, upper bound: 19342.7334477
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7336307, upper bound: 19342.7333181
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7764198, upper bound: 19342.7761389
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7742422, upper bound: 19342.7762727
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7559795
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7482020, upper bound: 19342.7559447
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6489372, upper bound: 19342.6568310
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6489372, upper bound: 19342.6568310
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7110019, upper bound: 19342.7111494
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7109275, upper bound: 19342.7113618
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5936306, upper bound: 19342.5921743
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5936306, upper bound: 19342.5921743
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6557880, upper bound: 19342.6555460
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6557166, upper bound: 19342.6555460
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6261584, upper bound: 19342.6261047
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6261584, upper bound: 19342.6261047
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7360715, upper bound: 19342.7362996
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7615835, upper bound: 19342.7617210
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7615903, upper bound: 19342.7617210
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7829909, upper bound: 19342.7851943
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7837732, upper bound: 19342.7848476
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7140007, upper bound: 19342.7133814
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7118810, upper bound: 19342.7138108
time: 0.86 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8848054, upper bound: 19342.8797762
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8792847, upper bound: 19342.8908113
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8881502, upper bound: 19342.8796721
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8881488, upper bound: 19342.8817691
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8582270, upper bound: 19342.8693423
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8690178
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7839750, upper bound: 19342.7916979
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7851263, upper bound: 19342.7916979
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8714615, upper bound: 19342.8561331
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8714615, upper bound: 19342.8561331
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8636767, upper bound: 19342.8636767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8731712, upper bound: 19342.8639491
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8481322, upper bound: 19342.8451236
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8399633, upper bound: 19342.8396467
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8396753, upper bound: 19342.8450505
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8451236
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7771784, upper bound: 19342.7761564
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7759832, upper bound: 19342.7759832
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7983496, upper bound: 19342.7981048
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7983496, upper bound: 19342.7981048
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.5378771, upper bound: 19342.5358883
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.5378771, upper bound: 19342.5358883
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7517201, upper bound: 19342.7516591
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7516591, upper bound: 19342.7519041
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7335290, upper bound: 19342.7334477
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7336307, upper bound: 19342.7333181
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7764198, upper bound: 19342.7761389
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7742422, upper bound: 19342.7762727
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7559795
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7482020, upper bound: 19342.7559447
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.6489372, upper bound: 19342.6568310
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.6489372, upper bound: 19342.6568310
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7110019, upper bound: 19342.7111494
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7109275, upper bound: 19342.7113618
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.5936306, upper bound: 19342.5921743
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.5936306, upper bound: 19342.5921743
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.6557880, upper bound: 19342.6555460
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.6557166, upper bound: 19342.6555460
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.6261584, upper bound: 19342.6261047
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.6261584, upper bound: 19342.6261047
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7360715, upper bound: 19342.7362996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7615835, upper bound: 19342.7617210
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7615903, upper bound: 19342.7617210
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7829909, upper bound: 19342.7851943
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7837732, upper bound: 19342.7848476
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7140007, upper bound: 19342.7133814
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 3, lower bound: -19342.7118810, upper bound: 19342.7138108

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8427324, upper bound: 19342.8424825
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8427324, upper bound: 19342.8422500
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8786582, upper bound: 19342.8904389
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8786367, upper bound: 19342.8903862
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8696782, upper bound: 19342.8608833
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8608833, upper bound: 19342.8613455
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8696782, upper bound: 19342.8612644
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8610004, upper bound: 19342.8629272
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7967012, upper bound: 19342.8041412
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7970490, upper bound: 19342.7964693
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8569073
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8690178
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7912098
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7912329
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7851263, upper bound: 19342.7916979
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7849942, upper bound: 19342.7849564
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 9

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8010463, upper bound: 19342.7877055
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8010463, upper bound: 19342.7875532
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8714372, upper bound: 19342.8559145
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8681564, upper bound: 19342.8559145
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8636767, upper bound: 19342.8636767
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8636767, upper bound: 19342.8636767
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8731712, upper bound: 19342.8636767
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8730887, upper bound: 19342.8639491
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8392228, upper bound: 19342.8451236
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8477730, upper bound: 19342.8444449
time: 1.19 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8393507, upper bound: 19342.8391280
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8385937, upper bound: 19342.8390297
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8396352, upper bound: 19342.8411509
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8391901, upper bound: 19342.8449583
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8450710
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8393802, upper bound: 19342.8451228
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6985152, upper bound: 19342.6985152
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6985152, upper bound: 19342.6985152
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6867543, upper bound: 19342.6867543
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6867543, upper bound: 19342.6867543
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7842561, upper bound: 19342.7826001
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7842561, upper bound: 19342.7826001
time: 0.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5963966, upper bound: 19342.5962286
time: 1.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5963966, upper bound: 19342.5962286
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7488220, upper bound: 19342.7488220
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7488282, upper bound: 19342.7488220
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7516575, upper bound: 19342.7517250
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7516575, upper bound: 19342.7518983
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5908778, upper bound: 19342.5926180
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.5908778, upper bound: 19342.5926180
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6821548, upper bound: 19342.6817235
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6821623, upper bound: 19342.6818460
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7043652, upper bound: 19342.7009968
time: 1.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7036053, upper bound: 19342.7038485
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7696580, upper bound: 19342.7762727
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7692285, upper bound: 19342.7761549
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7481282
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7517207
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7479292, upper bound: 19342.7482106
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7479201, upper bound: 19342.7556972
time: 1.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6474437, upper bound: 19342.6481393
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6470749, upper bound: 19342.6484079
time: 0.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7109275, upper bound: 19342.7109174
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7106206, upper bound: 19342.7113618
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7360715, upper bound: 19342.7357934
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7355571, upper bound: 19342.7362996
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6875308, upper bound: 19342.6869559
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6875308, upper bound: 19342.6869559
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6154287, upper bound: 19342.6154287
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6154287, upper bound: 19342.6154287
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7098932, upper bound: 19342.7098932
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7098932, upper bound: 19342.7098932
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7502237, upper bound: 19342.7506493
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7502304, upper bound: 19342.7506493
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7114993, upper bound: 19342.7125393
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7138222, upper bound: 19342.7131147
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6850032, upper bound: 19342.6855984
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6850032, upper bound: 19342.6850032
time: 0.93 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8427324, upper bound: 19342.8424825
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8427324, upper bound: 19342.8422500
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8786582, upper bound: 19342.8904389
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8786367, upper bound: 19342.8903862
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8696782, upper bound: 19342.8608833
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8608833, upper bound: 19342.8613455
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8696782, upper bound: 19342.8612644
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8610004, upper bound: 19342.8629272
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7967012, upper bound: 19342.8041412
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7970490, upper bound: 19342.7964693
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8569073
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8690178
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7912098
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7912329
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7851263, upper bound: 19342.7916979
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7849942, upper bound: 19342.7849564
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8010463, upper bound: 19342.7877055
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8010463, upper bound: 19342.7875532
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8714372, upper bound: 19342.8559145
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8681564, upper bound: 19342.8559145
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8636767, upper bound: 19342.8636767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8636767, upper bound: 19342.8636767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8731712, upper bound: 19342.8636767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8730887, upper bound: 19342.8639491
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8392228, upper bound: 19342.8451236
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8477730, upper bound: 19342.8444449
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8393507, upper bound: 19342.8391280
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8385937, upper bound: 19342.8390297
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8396352, upper bound: 19342.8411509
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8391901, upper bound: 19342.8449583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8450710
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.8393802, upper bound: 19342.8451228
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6985152, upper bound: 19342.6985152
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6985152, upper bound: 19342.6985152
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6867543, upper bound: 19342.6867543
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6867543, upper bound: 19342.6867543
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7842561, upper bound: 19342.7826001
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7842561, upper bound: 19342.7826001
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.5963966, upper bound: 19342.5962286
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.5963966, upper bound: 19342.5962286
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7488220, upper bound: 19342.7488220
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7488282, upper bound: 19342.7488220
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7516575, upper bound: 19342.7517250
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7516575, upper bound: 19342.7518983
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.5908778, upper bound: 19342.5926180
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.5908778, upper bound: 19342.5926180
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6821548, upper bound: 19342.6817235
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6821623, upper bound: 19342.6818460
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7043652, upper bound: 19342.7009968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7036053, upper bound: 19342.7038485
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7696580, upper bound: 19342.7762727
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7692285, upper bound: 19342.7761549
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7481282
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7517207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7479292, upper bound: 19342.7482106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7479201, upper bound: 19342.7556972
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6474437, upper bound: 19342.6481393
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6470749, upper bound: 19342.6484079
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7109275, upper bound: 19342.7109174
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7106206, upper bound: 19342.7113618
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7360715, upper bound: 19342.7357934
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7355571, upper bound: 19342.7362996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6875308, upper bound: 19342.6869559
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6875308, upper bound: 19342.6869559
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6154287, upper bound: 19342.6154287
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6154287, upper bound: 19342.6154287
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7098932, upper bound: 19342.7098932
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7098932, upper bound: 19342.7098932
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7502237, upper bound: 19342.7506493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7502304, upper bound: 19342.7506493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7114993, upper bound: 19342.7125393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.7138222, upper bound: 19342.7131147
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6850032, upper bound: 19342.6855984
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.76
Output dim: 3, lower bound: -19342.6850032, upper bound: 19342.6850032

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8419541, upper bound: 19342.8419541
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8427324, upper bound: 19342.8424825
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7793555, upper bound: 19342.7800770
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7791546, upper bound: 19342.7824850
time: 2.59 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8786367, upper bound: 19342.8904389
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8786582, upper bound: 19342.8901937
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8474402, upper bound: 19342.8483762
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8474402, upper bound: 19342.8474402
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8696501, upper bound: 19342.8608833
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8608833, upper bound: 19342.8608833
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8606597, upper bound: 19342.8606597
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8606597, upper bound: 19342.8611555
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8581403, upper bound: 19342.8568244
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8575985, upper bound: 19342.8568244
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8610004, upper bound: 19342.8629272
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8608833, upper bound: 19342.8625614
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7309534, upper bound: 19342.7321139
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7311043, upper bound: 19342.7311746
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7965256, upper bound: 19342.7962545
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7969152, upper bound: 19342.7962696
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8569073
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8569073
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8060762, upper bound: 19342.8064659
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8060762, upper bound: 19342.8060762
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7381171, upper bound: 19342.7417828
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7381171, upper bound: 19342.7460534
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7834377
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7912329
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7398130, upper bound: 19342.7417286
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7384531, upper bound: 19342.7465411
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7345562, upper bound: 19342.7377104
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7345562, upper bound: 19342.7377104
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7923941, upper bound: 19342.7861262
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7991193, upper bound: 19342.7859282
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7167229, upper bound: 19342.7167229
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7235714, upper bound: 19342.7167229
time: 0.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8364219, upper bound: 19342.8210974
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8210974, upper bound: 19342.8210974
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7740480, upper bound: 19342.7416056
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7740480, upper bound: 19342.7416056
time: 0.77 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8636690, upper bound: 19342.8636690
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8636690, upper bound: 19342.8636690
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7817588, upper bound: 19342.7815805
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7815805, upper bound: 19342.7815805
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8067685, upper bound: 19342.8038904
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8028285, upper bound: 19342.8028285
time: 0.92 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.21 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8419541, upper bound: 19342.8419541
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8427324, upper bound: 19342.8424825
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7793555, upper bound: 19342.7800770
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7791546, upper bound: 19342.7824850
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8786367, upper bound: 19342.8904389
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8786582, upper bound: 19342.8901937
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8474402, upper bound: 19342.8483762
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8474402, upper bound: 19342.8474402
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8696501, upper bound: 19342.8608833
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8608833, upper bound: 19342.8608833
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8606597, upper bound: 19342.8606597
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8606597, upper bound: 19342.8611555
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8581403, upper bound: 19342.8568244
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8575985, upper bound: 19342.8568244
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8610004, upper bound: 19342.8629272
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8608833, upper bound: 19342.8625614
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7309534, upper bound: 19342.7321139
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7311043, upper bound: 19342.7311746
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7965256, upper bound: 19342.7962545
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7969152, upper bound: 19342.7962696
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8569073
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8569073, upper bound: 19342.8569073
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8060762, upper bound: 19342.8064659
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8060762, upper bound: 19342.8060762
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7381171, upper bound: 19342.7417828
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7381171, upper bound: 19342.7460534
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7834377
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7834377, upper bound: 19342.7912329
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7398130, upper bound: 19342.7417286
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7384531, upper bound: 19342.7465411
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7345562, upper bound: 19342.7377104
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7345562, upper bound: 19342.7377104
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7923941, upper bound: 19342.7861262
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7991193, upper bound: 19342.7859282
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7167229, upper bound: 19342.7167229
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7235714, upper bound: 19342.7167229
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8364219, upper bound: 19342.8210974
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8210974, upper bound: 19342.8210974
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7740480, upper bound: 19342.7416056
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7740480, upper bound: 19342.7416056
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8636690, upper bound: 19342.8636690
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8636690, upper bound: 19342.8636690
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7817588, upper bound: 19342.7815805
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.7815805, upper bound: 19342.7815805
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8067685, upper bound: 19342.8038904
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.21
Output dim: 3, lower bound: -19342.8028285, upper bound: 19342.8028285
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8730887, upper bound: 19342.8639491
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8392228, upper bound: 19342.8451236
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8477730, upper bound: 19342.8444449
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8393507, upper bound: 19342.8391280
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8385937, upper bound: 19342.8390297
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8396352, upper bound: 19342.8411509
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8391901, upper bound: 19342.8449583
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8497682, upper bound: 19342.8450710
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.8393802, upper bound: 19342.8451228
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6985152, upper bound: 19342.6985152
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6985152, upper bound: 19342.6985152
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6867543, upper bound: 19342.6867543
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6867543, upper bound: 19342.6867543
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7842561, upper bound: 19342.7826001
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7842561, upper bound: 19342.7826001
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7488220, upper bound: 19342.7488220
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7488282, upper bound: 19342.7488220
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7516575, upper bound: 19342.7517250
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7516575, upper bound: 19342.7518983
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7043652, upper bound: 19342.7009968
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7036053, upper bound: 19342.7038485
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7696580, upper bound: 19342.7762727
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7692285, upper bound: 19342.7761549
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7481282
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7481282, upper bound: 19342.7517207
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7479292, upper bound: 19342.7482106
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7479201, upper bound: 19342.7556972
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7109275, upper bound: 19342.7109174
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7106206, upper bound: 19342.7113618
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7355563, upper bound: 19342.7355563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7360715, upper bound: 19342.7357934
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7355571, upper bound: 19342.7362996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6875308, upper bound: 19342.6869559
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6875308, upper bound: 19342.6869559
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7098932, upper bound: 19342.7098932
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7098932, upper bound: 19342.7098932
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7502237, upper bound: 19342.7506493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7502304, upper bound: 19342.7506493
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7114993, upper bound: 19342.7125393
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.7138222, upper bound: 19342.7131147
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6850032, upper bound: 19342.6855984
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.21
Output dim: 3, lower bound: -19342.6850032, upper bound: 19342.6850032
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=24289.171875
rel_dist={3: [-19343.13957058345, 19343.13957058345]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 16

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.1201403, upper bound: 19343.1187923
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.1187923, upper bound: 19343.1201403
time: 0.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.63 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 3, lower bound: -19343.1201403, upper bound: 19343.1187923
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.63
Output dim: 3, lower bound: -19343.1187923, upper bound: 19343.1201403

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.1084986, upper bound: 19343.1056471
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.1069501, upper bound: 19343.1072866
time: 0.89 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0992281, upper bound: 19343.0901818
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0896935, upper bound: 19343.1000116
time: 0.94 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.1084986, upper bound: 19343.1056471
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.1069501, upper bound: 19343.1072866
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.0992281, upper bound: 19343.0901818
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.0896935, upper bound: 19343.1000116

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0948655, upper bound: 19343.0912803
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0948130, upper bound: 19343.0912932
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0904256, upper bound: 19343.0909689
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0905765, upper bound: 19343.0904484
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0008943, upper bound: 19342.9953310
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0035304, upper bound: 19342.9953310
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8833217, upper bound: 19342.8852485
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8833217, upper bound: 19342.8852485
time: 0.84 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.70 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19343.0948655, upper bound: 19343.0912803
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19343.0948130, upper bound: 19343.0912932
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19343.0904256, upper bound: 19343.0909689
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19343.0905765, upper bound: 19343.0904484
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19343.0008943, upper bound: 19342.9953310
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19343.0035304, upper bound: 19342.9953310
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19342.8833217, upper bound: 19342.8852485
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.70
Output dim: 3, lower bound: -19342.8833217, upper bound: 19342.8852485

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0077818, upper bound: 19343.0081885
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0084664, upper bound: 19343.0081885
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9113078, upper bound: 19342.9072615
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9115921, upper bound: 19342.9052438
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0882581, upper bound: 19343.0899661
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0882581, upper bound: 19343.0899661
time: 0.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0903145, upper bound: 19343.0873635
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0879305, upper bound: 19343.0904332
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9784236, upper bound: 19342.9684214
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9794639, upper bound: 19342.9669018
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9887998, upper bound: 19342.9784332
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9888505, upper bound: 19342.9774652
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8701946, upper bound: 19342.8717336
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8685087, upper bound: 19342.8717062
time: 1.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7146729, upper bound: 19342.7191379
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7146729, upper bound: 19342.7191379
time: 0.93 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19343.0077818, upper bound: 19343.0081885
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19343.0084664, upper bound: 19343.0081885
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.9113078, upper bound: 19342.9072615
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.9115921, upper bound: 19342.9052438
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19343.0882581, upper bound: 19343.0899661
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19343.0882581, upper bound: 19343.0899661
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19343.0903145, upper bound: 19343.0873635
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19343.0879305, upper bound: 19343.0904332
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.9784236, upper bound: 19342.9684214
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.9794639, upper bound: 19342.9669018
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.9887998, upper bound: 19342.9784332
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.9888505, upper bound: 19342.9774652
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.8701946, upper bound: 19342.8717336
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.8685087, upper bound: 19342.8717062
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.7146729, upper bound: 19342.7191379
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.47
Output dim: 3, lower bound: -19342.7146729, upper bound: 19342.7191379

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8557700, upper bound: 19342.8572820
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8557700, upper bound: 19342.8572820
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0031153, upper bound: 19343.0025130
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0030476, upper bound: 19343.0026927
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8453432, upper bound: 19342.8415882
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8453037, upper bound: 19342.8416120
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9111319, upper bound: 19342.9052438
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9115921, upper bound: 19342.9049890
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0463152, upper bound: 19343.0447059
time: 1.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0445196, upper bound: 19343.0480046
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0882581, upper bound: 19343.0860999
time: 9.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0871912, upper bound: 19343.0883430
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0064388, upper bound: 19343.0064537
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0074391, upper bound: 19343.0061975
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0852153, upper bound: 19343.0881052
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0851125, upper bound: 19343.0881131
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9770547, upper bound: 19342.9672686
time: 1.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9770547, upper bound: 19342.9672686
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9705723, upper bound: 19342.9664146
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9792355, upper bound: 19342.9653036
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9850394, upper bound: 19342.9726882
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9852001, upper bound: 19342.9726882
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7467292, upper bound: 19342.7531300
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7467292, upper bound: 19342.7531300
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8701946, upper bound: 19342.8695563
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8696438, upper bound: 19342.8717336
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8572974, upper bound: 19342.8592203
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8573036, upper bound: 19342.8593137
time: 1.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6641833, upper bound: 19342.6658281
time: 1.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6644303, upper bound: 19342.6656430
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6927696, upper bound: 19342.6799845
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6826465, upper bound: 19342.6957489
time: 0.91 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8557700, upper bound: 19342.8572820
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8557700, upper bound: 19342.8572820
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0031153, upper bound: 19343.0025130
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0030476, upper bound: 19343.0026927
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8453432, upper bound: 19342.8415882
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8453037, upper bound: 19342.8416120
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9111319, upper bound: 19342.9052438
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9115921, upper bound: 19342.9049890
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0463152, upper bound: 19343.0447059
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0445196, upper bound: 19343.0480046
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0882581, upper bound: 19343.0860999
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0871912, upper bound: 19343.0883430
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0064388, upper bound: 19343.0064537
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0074391, upper bound: 19343.0061975
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0852153, upper bound: 19343.0881052
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19343.0851125, upper bound: 19343.0881131
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9770547, upper bound: 19342.9672686
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9770547, upper bound: 19342.9672686
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9705723, upper bound: 19342.9664146
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9792355, upper bound: 19342.9653036
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9850394, upper bound: 19342.9726882
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.9852001, upper bound: 19342.9726882
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.7467292, upper bound: 19342.7531300
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.7467292, upper bound: 19342.7531300
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8701946, upper bound: 19342.8695563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8696438, upper bound: 19342.8717336
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8572974, upper bound: 19342.8592203
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.8573036, upper bound: 19342.8593137
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.6641833, upper bound: 19342.6658281
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.6644303, upper bound: 19342.6656430
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.6927696, upper bound: 19342.6799845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.60
Output dim: 3, lower bound: -19342.6826465, upper bound: 19342.6957489

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4965798, upper bound: 19342.4967859
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4965798, upper bound: 19342.4965798
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7001437, upper bound: 19342.7001437
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7001437, upper bound: 19342.7001437
time: 0.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9765146, upper bound: 19342.9754948
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9765126, upper bound: 19342.9766886
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0008336, upper bound: 19342.9998793
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0028514, upper bound: 19343.0017478
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8351147, upper bound: 19342.8320929
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8346960, upper bound: 19342.8334379
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8404068, upper bound: 19342.8334103
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8403735, upper bound: 19342.8368106
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9015887, upper bound: 19342.8959688
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9015887, upper bound: 19342.8959688
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6780767, upper bound: 19342.6731462
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6780767, upper bound: 19342.6731462
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0433502, upper bound: 19343.0388306
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0407112, upper bound: 19343.0419178
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0441962, upper bound: 19343.0478933
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0443326, upper bound: 19343.0403980
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0310152, upper bound: 19343.0292267
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0309667, upper bound: 19343.0293363
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0821529, upper bound: 19343.0827997
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0788369, upper bound: 19343.0846317
time: 0.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9357071, upper bound: 19342.9358665
time: 1.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9357071, upper bound: 19342.9358665
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3097598, upper bound: 19342.3097598
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3097598, upper bound: 19342.3097598
time: 0.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3149297, upper bound: 19342.3159871
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3149297, upper bound: 19342.3159871
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8998980, upper bound: 19342.9032444
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8998980, upper bound: 19342.9001350
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8936560, upper bound: 19342.8930065
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8996435, upper bound: 19342.8919170
time: 1.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9727785, upper bound: 19342.9668571
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9763524, upper bound: 19342.9646942
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9705397, upper bound: 19342.9664146
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9705723, upper bound: 19342.9664075
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9781447, upper bound: 19342.9645582
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9658751, upper bound: 19342.9646584
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9777530, upper bound: 19342.9726882
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9850394, upper bound: 19342.9694117
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9846843, upper bound: 19342.9726882
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9848325, upper bound: 19342.9722230
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7467292, upper bound: 19342.7531300
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7463777, upper bound: 19342.7416250
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7166635, upper bound: 19342.7016067
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7155205, upper bound: 19342.7117204
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8278611, upper bound: 19342.8316797
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8291582, upper bound: 19342.8308563
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8695816, upper bound: 19342.8717336
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8696438, upper bound: 19342.8716695
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8063916, upper bound: 19342.8084560
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8063916, upper bound: 19342.8084560
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6059545, upper bound: 19342.6124592
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6059545, upper bound: 19342.6124592
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6736450, upper bound: 19342.6736450
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6927696, upper bound: 19342.6799845
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6681694, upper bound: 19342.6676406
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6681694, upper bound: 19342.6777951
time: 1.31 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 5.44 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.4965798, upper bound: 19342.4967859
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.4965798, upper bound: 19342.4965798
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.7001437, upper bound: 19342.7001437
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.7001437, upper bound: 19342.7001437
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9765146, upper bound: 19342.9754948
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9765126, upper bound: 19342.9766886
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0008336, upper bound: 19342.9998793
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0028514, upper bound: 19343.0017478
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8351147, upper bound: 19342.8320929
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8346960, upper bound: 19342.8334379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8404068, upper bound: 19342.8334103
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8403735, upper bound: 19342.8368106
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9015887, upper bound: 19342.8959688
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9015887, upper bound: 19342.8959688
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6780767, upper bound: 19342.6731462
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6780767, upper bound: 19342.6731462
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0433502, upper bound: 19343.0388306
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0407112, upper bound: 19343.0419178
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0441962, upper bound: 19343.0478933
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0443326, upper bound: 19343.0403980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0310152, upper bound: 19343.0292267
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0309667, upper bound: 19343.0293363
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0821529, upper bound: 19343.0827997
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19343.0788369, upper bound: 19343.0846317
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9357071, upper bound: 19342.9358665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9357071, upper bound: 19342.9358665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.3097598, upper bound: 19342.3097598
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.3097598, upper bound: 19342.3097598
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.3149297, upper bound: 19342.3159871
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.3149297, upper bound: 19342.3159871
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8998980, upper bound: 19342.9032444
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8998980, upper bound: 19342.9001350
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8936560, upper bound: 19342.8930065
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8996435, upper bound: 19342.8919170
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9727785, upper bound: 19342.9668571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9763524, upper bound: 19342.9646942
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9705397, upper bound: 19342.9664146
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9705723, upper bound: 19342.9664075
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9781447, upper bound: 19342.9645582
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9658751, upper bound: 19342.9646584
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9777530, upper bound: 19342.9726882
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9850394, upper bound: 19342.9694117
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9846843, upper bound: 19342.9726882
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.9848325, upper bound: 19342.9722230
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.7467292, upper bound: 19342.7531300
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.7463777, upper bound: 19342.7416250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.7166635, upper bound: 19342.7016067
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.7155205, upper bound: 19342.7117204
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8278611, upper bound: 19342.8316797
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8291582, upper bound: 19342.8308563
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8695816, upper bound: 19342.8717336
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8696438, upper bound: 19342.8716695
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8063916, upper bound: 19342.8084560
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.8063916, upper bound: 19342.8084560
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6059545, upper bound: 19342.6124592
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6059545, upper bound: 19342.6124592
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6736450, upper bound: 19342.6736450
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6927696, upper bound: 19342.6799845
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6681694, upper bound: 19342.6676406
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 5.44
Output dim: 3, lower bound: -19342.6681694, upper bound: 19342.6777951

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6269565, upper bound: 19342.6269565
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6269565, upper bound: 19342.6269565
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6980927, upper bound: 19342.6980927
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6980927, upper bound: 19342.6980927
time: 1.00 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9756211, upper bound: 19342.9754859
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9765146, upper bound: 19342.9754859
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9761536, upper bound: 19342.9766497
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9762074, upper bound: 19342.9755410
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9985658, upper bound: 19342.9976439
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9979436, upper bound: 19342.9976439
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9537697, upper bound: 19342.9533204
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9540103, upper bound: 19342.9535446
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8331801, upper bound: 19342.8305703
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8318361, upper bound: 19342.8296737
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8346960, upper bound: 19342.8334379
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8340236, upper bound: 19342.8318527
time: 0.83 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8329643, upper bound: 19342.8329643
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8404068, upper bound: 19342.8330767
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6413314, upper bound: 19342.6395004
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6413314, upper bound: 19342.6395004
time: 0.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8983897, upper bound: 19342.8929078
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8967244, upper bound: 19342.8929078
time: 1.06 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 19

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8753265, upper bound: 19342.8599022
time: 1.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8627282, upper bound: 19342.8599022
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0401726, upper bound: 19343.0387213
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0428628, upper bound: 19343.0387213
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0380949, upper bound: 19343.0382867
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0368896, upper bound: 19343.0384227
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0415677, upper bound: 19343.0473169
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0433671, upper bound: 19343.0450864
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6545009, upper bound: 19342.6545009
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6545009, upper bound: 19342.6545009
time: 2.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0012864, upper bound: 19342.9990064
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9983960, upper bound: 19342.9991133
time: 1.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8566034, upper bound: 19342.8533209
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8574150, upper bound: 19342.8533209
time: 1.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0733998, upper bound: 19343.0734108
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0733998, upper bound: 19343.0734108
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8611647, upper bound: 19342.8637417
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8611647, upper bound: 19342.8611647
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9356758, upper bound: 19342.9358357
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9356015, upper bound: 19342.9356015
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9357071, upper bound: 19342.9358665
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9356015, upper bound: 19342.9356015
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8909142, upper bound: 19342.8952538
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8906444, upper bound: 19342.8933680
time: 1.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8991948, upper bound: 19342.8991948
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8995705, upper bound: 19342.8997287
time: 1.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8895396, upper bound: 19342.8920765
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8929819, upper bound: 19342.8920685
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8994526, upper bound: 19342.8886533
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8895652, upper bound: 19342.8912840
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9727235, upper bound: 19342.9668571
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9641518, upper bound: 19342.9641518
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9732922, upper bound: 19342.9609046
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9632010, upper bound: 19342.9614218
time: 0.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9690364, upper bound: 19342.9653795
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9691232, upper bound: 19342.9652338
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 37

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9705723, upper bound: 19342.9664075
time: 0.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9696173, upper bound: 19342.9663519
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9455096, upper bound: 19342.9324172
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9457393, upper bound: 19342.9333499
time: 1.23 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9152086, upper bound: 19342.9131003
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9130090, upper bound: 19342.9130090
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9645829, upper bound: 19342.9572686
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9645506, upper bound: 19342.9545487
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9688474, upper bound: 19342.9632415
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9773230, upper bound: 19342.9641234
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9751813, upper bound: 19342.9709696
time: 1.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9834201, upper bound: 19342.9682310
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9810548, upper bound: 19342.9706040
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9834412, upper bound: 19342.9670689
time: 1.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 34

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7425943, upper bound: 19342.7372817
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7427708, upper bound: 19342.7477898
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7463777, upper bound: 19342.7416250
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7416250, upper bound: 19342.7416250
time: 1.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7001913, upper bound: 19342.6915209
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7063611, upper bound: 19342.6915209
time: 0.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6757969, upper bound: 19342.6757969
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6819615, upper bound: 19342.6757969
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7339327, upper bound: 19342.7356082
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7339327, upper bound: 19342.7356082
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8018308, upper bound: 19342.8016780
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8019361, upper bound: 19342.8016780
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8557756, upper bound: 19342.8693970
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8671985, upper bound: 19342.8693840
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8562571, upper bound: 19342.8703863
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8686501, upper bound: 19342.8641258
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8042091, upper bound: 19342.8062228
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8006160, upper bound: 19342.8075178
time: 0.85 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6269565, upper bound: 19342.6269565
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6269565, upper bound: 19342.6269565
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6980927, upper bound: 19342.6980927
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6980927, upper bound: 19342.6980927
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9756211, upper bound: 19342.9754859
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9765146, upper bound: 19342.9754859
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9761536, upper bound: 19342.9766497
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9762074, upper bound: 19342.9755410
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9985658, upper bound: 19342.9976439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9979436, upper bound: 19342.9976439
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9537697, upper bound: 19342.9533204
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9540103, upper bound: 19342.9535446
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8331801, upper bound: 19342.8305703
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8318361, upper bound: 19342.8296737
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8346960, upper bound: 19342.8334379
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8340236, upper bound: 19342.8318527
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8329643, upper bound: 19342.8329643
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8404068, upper bound: 19342.8330767
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6413314, upper bound: 19342.6395004
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6413314, upper bound: 19342.6395004
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8983897, upper bound: 19342.8929078
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8967244, upper bound: 19342.8929078
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8753265, upper bound: 19342.8599022
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8627282, upper bound: 19342.8599022
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0401726, upper bound: 19343.0387213
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0428628, upper bound: 19343.0387213
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0380949, upper bound: 19343.0382867
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0368896, upper bound: 19343.0384227
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0415677, upper bound: 19343.0473169
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0433671, upper bound: 19343.0450864
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6545009, upper bound: 19342.6545009
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6545009, upper bound: 19342.6545009
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0012864, upper bound: 19342.9990064
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9983960, upper bound: 19342.9991133
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8566034, upper bound: 19342.8533209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8574150, upper bound: 19342.8533209
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0733998, upper bound: 19343.0734108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19343.0733998, upper bound: 19343.0734108
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8611647, upper bound: 19342.8637417
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8611647, upper bound: 19342.8611647
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9356758, upper bound: 19342.9358357
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9356015, upper bound: 19342.9356015
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9357071, upper bound: 19342.9358665
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9356015, upper bound: 19342.9356015
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8909142, upper bound: 19342.8952538
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8906444, upper bound: 19342.8933680
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8991948, upper bound: 19342.8991948
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8995705, upper bound: 19342.8997287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8895396, upper bound: 19342.8920765
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8929819, upper bound: 19342.8920685
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8994526, upper bound: 19342.8886533
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8895652, upper bound: 19342.8912840
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9727235, upper bound: 19342.9668571
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9641518, upper bound: 19342.9641518
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9732922, upper bound: 19342.9609046
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9632010, upper bound: 19342.9614218
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9690364, upper bound: 19342.9653795
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9691232, upper bound: 19342.9652338
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9705723, upper bound: 19342.9664075
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9696173, upper bound: 19342.9663519
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9455096, upper bound: 19342.9324172
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9457393, upper bound: 19342.9333499
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9152086, upper bound: 19342.9131003
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9130090, upper bound: 19342.9130090
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9645829, upper bound: 19342.9572686
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9645506, upper bound: 19342.9545487
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9688474, upper bound: 19342.9632415
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9773230, upper bound: 19342.9641234
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9751813, upper bound: 19342.9709696
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9834201, upper bound: 19342.9682310
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9810548, upper bound: 19342.9706040
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.9834412, upper bound: 19342.9670689
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7425943, upper bound: 19342.7372817
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7427708, upper bound: 19342.7477898
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7463777, upper bound: 19342.7416250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7416250, upper bound: 19342.7416250
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7001913, upper bound: 19342.6915209
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7063611, upper bound: 19342.6915209
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6757969, upper bound: 19342.6757969
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.6819615, upper bound: 19342.6757969
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7339327, upper bound: 19342.7356082
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.7339327, upper bound: 19342.7356082
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8018308, upper bound: 19342.8016780
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8019361, upper bound: 19342.8016780
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8557756, upper bound: 19342.8693970
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8671985, upper bound: 19342.8693840
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8562571, upper bound: 19342.8703863
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8686501, upper bound: 19342.8641258
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8042091, upper bound: 19342.8062228
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.77
Output dim: 3, lower bound: -19342.8006160, upper bound: 19342.8075178
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -19342.8063916, upper bound: 19342.8084560
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.77
Output dim: 3, lower bound: -19342.6927696, upper bound: 19342.6799845
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=24289.171875
rel_dist={3: [-19343.120444796758, 19343.120444796754]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0782066, upper bound: 19343.0741153
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0741153, upper bound: 19343.0782066
time: 0.86 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.81 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 3, lower bound: -19343.0782066, upper bound: 19343.0741153
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.81
Output dim: 3, lower bound: -19343.0741153, upper bound: 19343.0782066

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0733117, upper bound: 19343.0734483
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0770262, upper bound: 19343.0670930
time: 0.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0692989, upper bound: 19343.0735756
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0662225, upper bound: 19343.0744104
time: 0.85 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.65 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.0733117, upper bound: 19343.0734483
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.0770262, upper bound: 19343.0670930
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.0692989, upper bound: 19343.0735756
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -19343.0662225, upper bound: 19343.0744104

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0721249, upper bound: 19343.0677570
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0659346, upper bound: 19343.0723902
time: 0.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0769138, upper bound: 19343.0670844
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0769138, upper bound: 19343.0670844
time: 0.91 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0686017, upper bound: 19343.0733316
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0686398, upper bound: 19343.0733316
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0053106, upper bound: 19343.0217104
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0079067, upper bound: 19343.0179795
time: 0.96 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.51 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0721249, upper bound: 19343.0677570
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0659346, upper bound: 19343.0723902
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0769138, upper bound: 19343.0670844
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0769138, upper bound: 19343.0670844
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0686017, upper bound: 19343.0733316
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0686398, upper bound: 19343.0733316
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0053106, upper bound: 19343.0217104
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 3, lower bound: -19343.0079067, upper bound: 19343.0179795

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0591236, upper bound: 19343.0538398
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0591467, upper bound: 19343.0536507
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8552179, upper bound: 19342.8578483
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8552179, upper bound: 19342.8578483
time: 0.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0719524, upper bound: 19343.0609435
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0719524, upper bound: 19343.0609435
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0614241, upper bound: 19343.0518082
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0613990, upper bound: 19343.0517096
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9589588, upper bound: 19342.9607528
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9589588, upper bound: 19342.9540330
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0677650, upper bound: 19343.0705651
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0677101, upper bound: 19343.0724869
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0053106, upper bound: 19343.0217104
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0047914, upper bound: 19343.0183568
time: 1.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 23

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8809705, upper bound: 19342.8909053
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8809705, upper bound: 19342.8909053
time: 1.00 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.91 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0591236, upper bound: 19343.0538398
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0591467, upper bound: 19343.0536507
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19342.8552179, upper bound: 19342.8578483
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19342.8552179, upper bound: 19342.8578483
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0719524, upper bound: 19343.0609435
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0719524, upper bound: 19343.0609435
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0614241, upper bound: 19343.0518082
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0613990, upper bound: 19343.0517096
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19342.9589588, upper bound: 19342.9607528
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19342.9589588, upper bound: 19342.9540330
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0677650, upper bound: 19343.0705651
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0677101, upper bound: 19343.0724869
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0053106, upper bound: 19343.0217104
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19343.0047914, upper bound: 19343.0183568
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19342.8809705, upper bound: 19342.8909053
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.91
Output dim: 3, lower bound: -19342.8809705, upper bound: 19342.8909053

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.62 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0434499, upper bound: 19343.0388821
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0458769, upper bound: 19343.0410026
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0589172, upper bound: 19343.0535978
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0579360, upper bound: 19343.0536252
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.79 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8525941, upper bound: 19342.8531612
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8505166, upper bound: 19342.8552240
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8387954, upper bound: 19342.8384886
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8387954, upper bound: 19342.8384886
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0550626, upper bound: 19343.0434708
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0562599, upper bound: 19343.0430483
time: 1.02 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 31

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0582323, upper bound: 19343.0464400
time: 1.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0584474, upper bound: 19343.0464385
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0582380, upper bound: 19343.0497974
time: 1.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0600447, upper bound: 19343.0466006
time: 1.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0242944, upper bound: 19343.0195234
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0274030, upper bound: 19343.0178096
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9134615, upper bound: 19342.9172423
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9134615, upper bound: 19342.9172423
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9580705, upper bound: 19342.9529984
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9589588, upper bound: 19342.9540086
time: 0.89 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 47

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0324289, upper bound: 19343.0297408
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0315930, upper bound: 19343.0335439
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9119416, upper bound: 19342.9124497
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9119416, upper bound: 19342.9124497
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 36

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9927119, upper bound: 19343.0210259
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0047774, upper bound: 19343.0094751
time: 1.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9943464, upper bound: 19343.0183568
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0047914, upper bound: 19343.0181715
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8635577, upper bound: 19342.8751069
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8716029, upper bound: 19342.8787179
time: 0.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8346470, upper bound: 19342.8444298
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8361744, upper bound: 19342.8180262
time: 0.85 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.48 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0434499, upper bound: 19343.0388821
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0458769, upper bound: 19343.0410026
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0589172, upper bound: 19343.0535978
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0579360, upper bound: 19343.0536252
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8525941, upper bound: 19342.8531612
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8505166, upper bound: 19342.8552240
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8387954, upper bound: 19342.8384886
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8387954, upper bound: 19342.8384886
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0550626, upper bound: 19343.0434708
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0562599, upper bound: 19343.0430483
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0582323, upper bound: 19343.0464400
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0584474, upper bound: 19343.0464385
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0582380, upper bound: 19343.0497974
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0600447, upper bound: 19343.0466006
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0242944, upper bound: 19343.0195234
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0274030, upper bound: 19343.0178096
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9134615, upper bound: 19342.9172423
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9134615, upper bound: 19342.9172423
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9580705, upper bound: 19342.9529984
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9589588, upper bound: 19342.9540086
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0324289, upper bound: 19343.0297408
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0315930, upper bound: 19343.0335439
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9119416, upper bound: 19342.9124497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9119416, upper bound: 19342.9124497
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9927119, upper bound: 19343.0210259
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0047774, upper bound: 19343.0094751
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.9943464, upper bound: 19343.0183568
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19343.0047914, upper bound: 19343.0181715
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8635577, upper bound: 19342.8751069
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8716029, upper bound: 19342.8787179
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8346470, upper bound: 19342.8444298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -19342.8361744, upper bound: 19342.8180262

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9151857, upper bound: 19342.9167243
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9151881, upper bound: 19342.9167243
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0151257, upper bound: 19343.0107392
time: 1.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0151257, upper bound: 19343.0107392
time: 1.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0411946, upper bound: 19343.0432361
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0550400, upper bound: 19343.0432304
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 18

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0514215, upper bound: 19343.0430379
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0489690, upper bound: 19343.0449820
time: 0.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8065618, upper bound: 19342.8046739
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8065618, upper bound: 19342.8038799
time: 0.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8490048, upper bound: 19342.8498427
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8497304, upper bound: 19342.8549594
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8187258, upper bound: 19342.8077483
time: 0.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8187258, upper bound: 19342.8077483
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8184822, upper bound: 19342.8198546
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8193382, upper bound: 19342.8192783
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 32

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0461019, upper bound: 19343.0427006
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0550626, upper bound: 19343.0434708
time: 0.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0462967, upper bound: 19343.0427826
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0439127, upper bound: 19343.0425672
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0419354, upper bound: 19343.0295765
time: 1.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0429980, upper bound: 19343.0291325
time: 1.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 22

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697917, upper bound: 19342.7651063
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697917, upper bound: 19342.7651063
time: 0.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0557144, upper bound: 19343.0465716
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0582380, upper bound: 19343.0496506
time: 0.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0581355, upper bound: 19343.0466006
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0600447, upper bound: 19343.0463440
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9415806, upper bound: 19342.9385997
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9415806, upper bound: 19342.9385997
time: 1.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 0

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0232146, upper bound: 19343.0137078
time: 0.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0154415, upper bound: 19343.0136478
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9134615, upper bound: 19342.9086295
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9049936, upper bound: 19342.9172423
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 17

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7590024, upper bound: 19342.7676287
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7590024, upper bound: 19342.7676287
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7661269, upper bound: 19342.7456589
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7661269, upper bound: 19342.7456589
time: 0.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 43

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9284698, upper bound: 19342.9265426
time: 1.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9294762, upper bound: 19342.9259982
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 16

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8519967, upper bound: 19342.8261436
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8519967, upper bound: 19342.8261436
time: 0.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6276716, upper bound: 19342.6385061
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6276716, upper bound: 19342.6385061
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 23

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9118425, upper bound: 19342.9124497
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9013685, upper bound: 19342.9042561
time: 1.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 41

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 29

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9117547, upper bound: 19342.9000864
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8986322, upper bound: 19342.9124497
time: 1.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9256081, upper bound: 19342.9432274
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9256081, upper bound: 19342.9432292
time: 1.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0043401, upper bound: 19342.9928245
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0043401, upper bound: 19342.9978355
time: 1.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9267867, upper bound: 19342.9404628
time: 0.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9267867, upper bound: 19342.9404628
time: 1.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0047018, upper bound: 19343.0043245
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9939248, upper bound: 19343.0180977
time: 0.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8634295, upper bound: 19342.8751069
time: 0.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8635577, upper bound: 19342.8714178
time: 1.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4110636, upper bound: 19342.4110636
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4110636, upper bound: 19342.4110636
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 39

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8346470, upper bound: 19342.8444298
time: 1.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8206703, upper bound: 19342.8386301
time: 0.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7837191, upper bound: 19342.7775014
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7775014, upper bound: 19342.7775014
time: 0.89 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.72 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9151857, upper bound: 19342.9167243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9151881, upper bound: 19342.9167243
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0151257, upper bound: 19343.0107392
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0151257, upper bound: 19343.0107392
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0411946, upper bound: 19343.0432361
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0550400, upper bound: 19343.0432304
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0514215, upper bound: 19343.0430379
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0489690, upper bound: 19343.0449820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8065618, upper bound: 19342.8046739
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8065618, upper bound: 19342.8038799
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8490048, upper bound: 19342.8498427
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8497304, upper bound: 19342.8549594
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8187258, upper bound: 19342.8077483
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8187258, upper bound: 19342.8077483
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8184822, upper bound: 19342.8198546
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8193382, upper bound: 19342.8192783
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0461019, upper bound: 19343.0427006
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0550626, upper bound: 19343.0434708
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0462967, upper bound: 19343.0427826
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0439127, upper bound: 19343.0425672
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0419354, upper bound: 19343.0295765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0429980, upper bound: 19343.0291325
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7697917, upper bound: 19342.7651063
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7697917, upper bound: 19342.7651063
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0557144, upper bound: 19343.0465716
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0582380, upper bound: 19343.0496506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0581355, upper bound: 19343.0466006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0600447, upper bound: 19343.0463440
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9415806, upper bound: 19342.9385997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9415806, upper bound: 19342.9385997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0232146, upper bound: 19343.0137078
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0154415, upper bound: 19343.0136478
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9134615, upper bound: 19342.9086295
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9049936, upper bound: 19342.9172423
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7590024, upper bound: 19342.7676287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7590024, upper bound: 19342.7676287
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7661269, upper bound: 19342.7456589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7661269, upper bound: 19342.7456589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9284698, upper bound: 19342.9265426
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9294762, upper bound: 19342.9259982
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8519967, upper bound: 19342.8261436
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8519967, upper bound: 19342.8261436
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.6276716, upper bound: 19342.6385061
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.6276716, upper bound: 19342.6385061
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9118425, upper bound: 19342.9124497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9013685, upper bound: 19342.9042561
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9117547, upper bound: 19342.9000864
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8986322, upper bound: 19342.9124497
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9256081, upper bound: 19342.9432274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9256081, upper bound: 19342.9432292
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0043401, upper bound: 19342.9928245
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0043401, upper bound: 19342.9978355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9267867, upper bound: 19342.9404628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9267867, upper bound: 19342.9404628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19343.0047018, upper bound: 19343.0043245
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.9939248, upper bound: 19343.0180977
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8634295, upper bound: 19342.8751069
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8635577, upper bound: 19342.8714178
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.4110636, upper bound: 19342.4110636
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.4110636, upper bound: 19342.4110636
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8346470, upper bound: 19342.8444298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.8206703, upper bound: 19342.8386301
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7837191, upper bound: 19342.7775014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.72
Output dim: 3, lower bound: -19342.7775014, upper bound: 19342.7775014

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7891645, upper bound: 19342.7902288
time: 0.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7891645, upper bound: 19342.7902288
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.74 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 21

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7650320, upper bound: 19342.7662308
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7651868, upper bound: 19342.7658624
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 42

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6798072, upper bound: 19342.6829746
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6798072, upper bound: 19342.6829746
time: 0.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9337970, upper bound: 19342.9376976
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9389029, upper bound: 19342.9352979
time: 0.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 9

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7532212, upper bound: 19342.7532212
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7532212, upper bound: 19342.7532212
time: 0.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 18
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 32

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9819036, upper bound: 19342.9806100
time: 1.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9843570, upper bound: 19342.9787053
time: 0.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.73 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0357390, upper bound: 19343.0367016
time: 0.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0487677, upper bound: 19343.0366226
time: 1.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 32
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 24

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8654593, upper bound: 19342.8609499
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8655673, upper bound: 19342.8607643
time: 0.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -15979.9384766, 11824.5693359, -15979.9384766, 11824.5693359, -27804.5078125, 27804.5078125
1: -12933.5390625, 11445.0302734, -12933.5390625, 11445.0302734, -24378.5664062, 24378.5664062
2: -18861.8515625, 12502.3134766, -18861.8515625, 12502.3134766, -31364.1640625, 31364.1640625
3: -6526.2768555, 17762.8984375, -6526.2768555, 17762.8984375, -24289.1718750, 24289.1718750
4: -20728.6542969, 12361.6982422, -20728.6542969, 12361.6982422, -33090.3515625, 33090.3515625

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 41
type: RSZ, layer: 1, pos: 31
type: RSZ, layer: 1, pos: 24
type: RSZ, layer: 1, pos: 23
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 34
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 28
type: RSZ, layer: 1, pos: 8
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 17
type: RSZ, layer: 1, pos: 9
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 14
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 22
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 19
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 47
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 39
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 43
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 41

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8065618, upper bound: 19342.8046739
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8038799, upper bound: 19342.8038974
time: 0.93 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.81 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.7891645, upper bound: 19342.7902288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.7891645, upper bound: 19342.7902288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.7650320, upper bound: 19342.7662308
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.7651868, upper bound: 19342.7658624
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.6798072, upper bound: 19342.6829746
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.6798072, upper bound: 19342.6829746
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.9337970, upper bound: 19342.9376976
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.9389029, upper bound: 19342.9352979
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.7532212, upper bound: 19342.7532212
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.7532212, upper bound: 19342.7532212
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.9819036, upper bound: 19342.9806100
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.9843570, upper bound: 19342.9787053
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19343.0357390, upper bound: 19343.0367016
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19343.0487677, upper bound: 19343.0366226
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.8654593, upper bound: 19342.8609499
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.8655673, upper bound: 19342.8607643
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.8065618, upper bound: 19342.8046739
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.81
Output dim: 3, lower bound: -19342.8038799, upper bound: 19342.8038974
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8065618, upper bound: 19342.8038799
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8490048, upper bound: 19342.8498427
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8497304, upper bound: 19342.8549594
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8187258, upper bound: 19342.8077483
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8187258, upper bound: 19342.8077483
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8184822, upper bound: 19342.8198546
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8193382, upper bound: 19342.8192783
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0461019, upper bound: 19343.0427006
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0550626, upper bound: 19343.0434708
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0462967, upper bound: 19343.0427826
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0439127, upper bound: 19343.0425672
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0419354, upper bound: 19343.0295765
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0429980, upper bound: 19343.0291325
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7697917, upper bound: 19342.7651063
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7697917, upper bound: 19342.7651063
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0557144, upper bound: 19343.0465716
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0582380, upper bound: 19343.0496506
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0581355, upper bound: 19343.0466006
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0600447, upper bound: 19343.0463440
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9415806, upper bound: 19342.9385997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9415806, upper bound: 19342.9385997
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0232146, upper bound: 19343.0137078
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0154415, upper bound: 19343.0136478
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9134615, upper bound: 19342.9086295
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9049936, upper bound: 19342.9172423
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7590024, upper bound: 19342.7676287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7590024, upper bound: 19342.7676287
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7661269, upper bound: 19342.7456589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7661269, upper bound: 19342.7456589
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9284698, upper bound: 19342.9265426
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9294762, upper bound: 19342.9259982
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8519967, upper bound: 19342.8261436
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8519967, upper bound: 19342.8261436
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9118425, upper bound: 19342.9124497
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9013685, upper bound: 19342.9042561
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9117547, upper bound: 19342.9000864
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8986322, upper bound: 19342.9124497
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9256081, upper bound: 19342.9432274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9256081, upper bound: 19342.9432292
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0043401, upper bound: 19342.9928245
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0043401, upper bound: 19342.9978355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9267867, upper bound: 19342.9404628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9267867, upper bound: 19342.9404628
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19343.0047018, upper bound: 19343.0043245
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.9939248, upper bound: 19343.0180977
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8634295, upper bound: 19342.8751069
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8635577, upper bound: 19342.8714178
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8346470, upper bound: 19342.8444298
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.8206703, upper bound: 19342.8386301
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7837191, upper bound: 19342.7775014
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.81
Output dim: 3, lower bound: -19342.7775014, upper bound: 19342.7775014
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=24289.171875
rel_dist={3: [-19343.089144719655, 19343.089144719655]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 1116.55 seconds
