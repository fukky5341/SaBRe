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
execution time: IAR + LP analysis = 2.47 + 2.42 = 4.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -19343.1395978, upper bound: 19343.1395978


# Binary Search by BASE starts (time budget: 1195.10 seconds, max iter: 100)

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
Binary search time: 95.75 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1099.35 seconds

## Binary search (step 0) starts
Candidate diff: 0.5000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9143643, upper bound: 19343.0890293
time: 0.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.91 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 3, lower bound: -19342.9143643, upper bound: 19343.0890293
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.91
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -15606.2392578, 11546.4335938, -15979.9384766, 11824.5693359, -27430.8085938, 27526.3710938
1: -12631.3779297, 11177.4941406, -12933.5390625, 11445.0302734, -24076.4042969, 24111.0332031
2: -18419.4785156, 12209.6884766, -18861.8515625, 12502.3134766, -30921.7929688, 31071.5371094
3: -6374.2680664, 17344.5468750, -6526.2768555, 17762.8984375, -24137.1660156, 23870.8183594
4: -20242.6933594, 12071.0019531, -20728.6542969, 12361.6982422, -32604.3867188, 32799.6562500

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
time: 0.86 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
time: 0.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -17825.7265625, 13106.9628906, -15919.6386719, 11779.1074219, -29604.8339844, 29026.5976562
1: -14429.7958984, 12691.0000000, -12884.7705078, 11401.0341797, -25830.8300781, 25575.7695312
2: -21053.4453125, 13860.3105469, -18790.8925781, 12454.1416016, -33507.5859375, 32651.2031250
3: -7247.8081055, 19801.2207031, -6501.2280273, 17695.5605469, -24943.3691406, 26302.4472656
4: -23130.5800781, 13698.3564453, -20650.6757812, 12313.9687500, -35444.5429688, 34349.0312500

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
time: 0.87 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
time: 1.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.76 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.76
Output dim: 3, lower bound: -19342.9038512, upper bound: 19342.9038512

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -15606.2392578, 11546.4335938, -15606.2392578, 11546.4335938, -27152.6718750, 27152.6699219
1: -12631.3779297, 11177.4941406, -12631.3779297, 11177.4941406, -23808.8710938, 23808.8710938
2: -18419.4785156, 12209.6884766, -18419.4785156, 12209.6884766, -30629.1640625, 30629.1660156
3: -6374.2680664, 17344.5468750, -6374.2680664, 17344.5468750, -23718.8125000, 23718.8144531
4: -20242.6933594, 12071.0019531, -20242.6933594, 12071.0019531, -32313.6914062, 32313.6933594

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8932694, upper bound: 19343.0325042
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8920905, upper bound: 19342.9778274
time: 1.03 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -15606.2392578, 11546.4335938, -17825.7265625, 13106.9628906, -28713.1992188, 29372.1562500
1: -12631.3779297, 11177.4941406, -14429.7958984, 12691.0000000, -25322.3769531, 25607.2890625
2: -18419.4785156, 12209.6884766, -21053.4453125, 13860.3105469, -32279.7871094, 33263.1328125
3: -6374.2680664, 17344.5468750, -7247.8081055, 19801.2207031, -26175.4882812, 24592.3515625
4: -20242.6933594, 12071.0019531, -23130.5800781, 13698.3564453, -33941.0507812, 35201.5703125

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8932694, upper bound: 19343.0325042
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8920905, upper bound: 19342.9778274
time: 0.93 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -17825.7265625, 13106.9628906, -15606.2392578, 11546.4335938, -29372.1562500, 28713.1992188
1: -14429.7958984, 12691.0000000, -12631.3779297, 11177.4941406, -25607.2890625, 25322.3769531
2: -21053.4453125, 13860.3105469, -18419.4785156, 12209.6884766, -33263.1328125, 32279.7871094
3: -7247.8081055, 19801.2207031, -6374.2680664, 17344.5468750, -24592.3515625, 26175.4882812
4: -23130.5800781, 13698.3564453, -20242.6933594, 12071.0019531, -35201.5703125, 33941.0507812

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7841812, upper bound: 19342.5075083
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8830255, upper bound: 19342.8830255
time: 0.87 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -17825.7265625, 13106.9628906, -17825.7265625, 13106.9628906, -30932.6875000, 30932.6855469
1: -14429.7958984, 12691.0000000, -14429.7958984, 12691.0000000, -27120.7968750, 27120.7968750
2: -21053.4453125, 13860.3105469, -21053.4453125, 13860.3105469, -34913.7539062, 34913.7539062
3: -7247.8081055, 19801.2207031, -7247.8081055, 19801.2207031, -27049.0273438, 27049.0273438
4: -23130.5800781, 13698.3564453, -23130.5800781, 13698.3564453, -36828.9335938, 36828.9335938

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7841812, upper bound: 19342.5075083
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8830255, upper bound: 19342.8830255
time: 0.97 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.16 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.8932694, upper bound: 19343.0325042
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.8920905, upper bound: 19342.9778274
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.8932694, upper bound: 19343.0325042
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.8920905, upper bound: 19342.9778274
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.7841812, upper bound: 19342.5075083
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.8830255, upper bound: 19342.8830255
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.7841812, upper bound: 19342.5075083
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.16
Output dim: 3, lower bound: -19342.8830255, upper bound: 19342.8830255

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -15606.2392578, 11546.4335938, -26931.5273438, 26989.0449219
1: -12452.2431641, 11018.8076172, -12631.3779297, 11177.4941406, -23629.7382812, 23650.1855469
2: -18158.9550781, 12037.1357422, -18419.4785156, 12209.6884766, -30368.6425781, 30456.6132812
3: -6284.6875000, 17098.0156250, -6374.2680664, 17344.5468750, -23629.2324219, 23472.2832031
4: -19956.5410156, 11899.9658203, -20242.6933594, 12071.0019531, -32027.5429688, 32142.6562500

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
time: 2.17 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -15570.2441406, 11520.3779297, -27648.0664062, 27540.4941406
1: -13054.4599609, 11591.3388672, -12602.1630859, 11152.4160156, -24206.8750000, 24193.5019531
2: -19041.2832031, 12659.5800781, -18377.4003906, 12182.4423828, -31223.7265625, 31036.9765625
3: -6605.4584961, 17970.6660156, -6359.9487305, 17305.1289062, -23910.5878906, 24330.6132812
4: -20919.1621094, 12504.9296875, -20196.3242188, 12043.8886719, -32963.0468750, 32701.2539062

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -17825.7265625, 13106.9628906, -28492.0546875, 29208.5312500
1: -12452.2431641, 11018.8076172, -14429.7958984, 12691.0000000, -25143.2402344, 25448.6035156
2: -18158.9550781, 12037.1357422, -21053.4453125, 13860.3105469, -32019.2636719, 33090.5820312
3: -6284.6875000, 17098.0156250, -7247.8081055, 19801.2207031, -26085.9062500, 24345.8222656
4: -19956.5410156, 11899.9658203, -23130.5800781, 13698.3564453, -33654.8984375, 35030.5390625

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8563594
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9778274
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -17791.2070312, 13082.0292969, -29209.7148438, 29761.4550781
1: -13054.4599609, 11591.3388672, -14401.8310547, 12667.0390625, -25721.4980469, 25993.1699219
2: -19041.2832031, 12659.5800781, -21013.1230469, 13834.2392578, -32875.5234375, 33672.7031250
3: -6605.4584961, 17970.6660156, -7233.9619141, 19763.3300781, -26368.7890625, 25204.6289062
4: -20919.1621094, 12504.9296875, -23086.0585938, 13672.4355469, -34591.5976562, 35590.9882812

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8593958
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9778274
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -17610.9257812, 12945.5888672, -15606.2392578, 11546.4335938, -29157.3574219, 28551.8281250
1: -14255.5488281, 12535.5800781, -12631.3779297, 11177.4941406, -25433.0429688, 25166.9531250
2: -20799.7832031, 13690.3691406, -18419.4785156, 12209.6884766, -33009.4687500, 32109.8476562
3: -7160.5898438, 19561.6269531, -6374.2680664, 17344.5468750, -24505.1347656, 25935.8945312
4: -22852.2402344, 13529.6064453, -20242.6933594, 12071.0019531, -34923.2343750, 33772.2929688

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18657.2773438, 13784.1298828, -15570.2441406, 11520.3779297, -30177.6562500, 29354.3750000
1: -15102.3857422, 13346.4589844, -12602.1630859, 11152.4160156, -26254.8007812, 25948.6171875
2: -22039.4648438, 14576.5332031, -18377.4003906, 12182.4423828, -34221.9062500, 32953.9335938
3: -7609.1791992, 20777.9628906, -6359.9487305, 17305.1289062, -24914.3085938, 27137.9082031
4: -24205.4062500, 14394.3369141, -20196.3242188, 12043.8886719, -36249.2929688, 34590.6601562

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9721882, upper bound: 19342.8177835
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
time: 1.24 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17610.9257812, 12945.5888672, -17825.7265625, 13106.9628906, -30717.8847656, 30771.3164062
1: -14255.5488281, 12535.5800781, -14429.7958984, 12691.0000000, -26946.5468750, 26965.3750000
2: -20799.7832031, 13690.3691406, -21053.4453125, 13860.3105469, -34660.0937500, 34743.8085938
3: -7160.5898438, 19561.6269531, -7247.8081055, 19801.2207031, -26961.8085938, 26809.4316406
4: -22852.2402344, 13529.6064453, -23130.5800781, 13698.3564453, -36550.5976562, 36660.1796875

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4392977, upper bound: 19342.4392977
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4392977, upper bound: 19342.5075083
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18657.2773438, 13784.1298828, -17791.2070312, 13082.0292969, -31739.3027344, 31575.3359375
1: -15102.3857422, 13346.4589844, -14401.8310547, 12667.0390625, -27769.4179688, 27748.2871094
2: -22039.4648438, 14576.5332031, -21013.1230469, 13834.2392578, -35873.7031250, 35589.6562500
3: -7609.1791992, 20777.9628906, -7233.9619141, 19763.3300781, -27372.5097656, 28011.9257812
4: -24205.4062500, 14394.3369141, -23086.0585938, 13672.4355469, -37877.8437500, 37480.3945312

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8770391, upper bound: 19342.8106668
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
time: 0.94 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.95 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871575
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8563594
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9778274
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8593958
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9778274
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.4392977, upper bound: 19342.4392977
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.4392977, upper bound: 19342.5075083
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8770391, upper bound: 19342.8106668
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.95
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -15385.0976562, 11382.8056641, -26767.9003906, 26767.9023438
1: -12452.2431641, 11018.8076172, -12452.2431641, 11018.8076172, -23471.0507812, 23471.0507812
2: -18158.9550781, 12037.1357422, -18158.9550781, 12037.1357422, -30196.0898438, 30196.0898438
3: -6284.6875000, 17098.0156250, -6284.6875000, 17098.0156250, -23382.7031250, 23382.7031250
4: -19956.5410156, 11899.9658203, -19956.5410156, 11899.9658203, -31856.5078125, 31856.5058594

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418450
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -16127.6875000, 11970.2519531, -27355.3437500, 27510.4921875
1: -12452.2431641, 11018.8076172, -13054.4599609, 11591.3388672, -24043.5800781, 24073.2675781
2: -18158.9550781, 12037.1357422, -19041.2832031, 12659.5800781, -30818.5332031, 31078.4179688
3: -6284.6875000, 17098.0156250, -6605.4584961, 17970.6660156, -24255.3535156, 23703.4746094
4: -19956.5410156, 11899.9658203, -20919.1621094, 12504.9296875, -32461.4707031, 32819.1289062

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418450
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -15385.0976562, 11382.8056641, -27510.4921875, 27355.3437500
1: -13054.4599609, 11591.3388672, -12452.2431641, 11018.8076172, -24073.2675781, 24043.5800781
2: -19041.2832031, 12659.5800781, -18158.9550781, 12037.1357422, -31078.4179688, 30818.5332031
3: -6605.4584961, 17970.6660156, -6284.6875000, 17098.0156250, -23703.4746094, 24255.3535156
4: -20919.1621094, 12504.9296875, -19956.5410156, 11899.9658203, -32819.1250000, 32461.4707031

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871412
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -16127.6875000, 11970.2519531, -28097.9375000, 28097.9375000
1: -13054.4599609, 11591.3388672, -13054.4599609, 11591.3388672, -24645.7988281, 24645.7988281
2: -19041.2832031, 12659.5800781, -19041.2832031, 12659.5800781, -31700.8613281, 31700.8632812
3: -6605.4584961, 17970.6660156, -6605.4584961, 17970.6660156, -24576.1250000, 24576.1250000
4: -20919.1621094, 12504.9296875, -20919.1621094, 12504.9296875, -33424.0937500, 33424.0898438

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871412
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -17610.9257812, 12945.5888672, -28330.6875000, 28993.7304688
1: -12452.2431641, 11018.8076172, -14255.5488281, 12535.5800781, -24987.8183594, 25274.3554688
2: -18158.9550781, 12037.1357422, -20799.7832031, 13690.3691406, -31849.3242188, 32836.9179688
3: -6284.6875000, 17098.0156250, -7160.5898438, 19561.6269531, -25846.3125000, 24258.6035156
4: -19956.5410156, 11899.9658203, -22852.2402344, 13529.6064453, -33486.1367188, 34752.2031250

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2522304, upper bound: 19342.7864284
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.6841296
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -18657.2773438, 13784.1298828, -29169.2246094, 30040.0820312
1: -12452.2431641, 11018.8076172, -15102.3857422, 13346.4589844, -25798.6953125, 26121.1933594
2: -18158.9550781, 12037.1357422, -22039.4648438, 14576.5332031, -32735.4882812, 34076.6015625
3: -6284.6875000, 17098.0156250, -7609.1791992, 20777.9628906, -27062.6484375, 24707.1953125
4: -19956.5410156, 11899.9658203, -24205.4062500, 14394.3369141, -34350.8750000, 36105.3710938

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2522304, upper bound: 19343.0269200
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.8680237
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -17610.9257812, 12945.5888672, -29073.2773438, 29581.1738281
1: -13054.4599609, 11591.3388672, -14255.5488281, 12535.5800781, -25590.0390625, 25846.8867188
2: -19041.2832031, 12659.5800781, -20799.7832031, 13690.3691406, -32731.6503906, 33459.3632812
3: -6605.4584961, 17970.6660156, -7160.5898438, 19561.6269531, -26167.0859375, 25131.2558594
4: -20919.1621094, 12504.9296875, -22852.2402344, 13529.6064453, -34448.7617188, 35357.1718750

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2507117, upper bound: 19342.7389709
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8593958
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8593958
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -18657.2773438, 13784.1298828, -29911.8164062, 30627.5273438
1: -13054.4599609, 11591.3388672, -15102.3857422, 13346.4589844, -26400.9160156, 26693.7226562
2: -19041.2832031, 12659.5800781, -22039.4648438, 14576.5332031, -33617.8125000, 34699.0429688
3: -6605.4584961, 17970.6660156, -7609.1791992, 20777.9628906, -27383.4218750, 25579.8457031
4: -20919.1621094, 12504.9296875, -24205.4062500, 14394.3369141, -35313.4921875, 36710.3359375

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2507117, upper bound: 19342.9277785
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9528033
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9528033
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -17610.9257812, 12945.5888672, -15385.0976562, 11382.8056641, -28993.7304688, 28330.6875000
1: -14255.5488281, 12535.5800781, -12452.2431641, 11018.8076172, -25274.3554688, 24987.8183594
2: -20799.7832031, 13690.3691406, -18158.9550781, 12037.1357422, -32836.9179688, 31849.3242188
3: -7160.5898438, 19561.6269531, -6284.6875000, 17098.0156250, -24258.6054688, 25846.3105469
4: -22852.2402344, 13529.6064453, -19956.5410156, 11899.9658203, -34752.2031250, 33486.1406250

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6259932, upper bound: 19341.9917611
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -17610.9257812, 12945.5888672, -16127.6875000, 11970.2519531, -29581.1738281, 29073.2773438
1: -14255.5488281, 12535.5800781, -13054.4599609, 11591.3388672, -25846.8867188, 25590.0390625
2: -20799.7832031, 13690.3691406, -19041.2832031, 12659.5800781, -33459.3632812, 32731.6523438
3: -7160.5898438, 19561.6269531, -6605.4584961, 17970.6660156, -25131.2558594, 26167.0859375
4: -22852.2402344, 13529.6064453, -20919.1621094, 12504.9296875, -35357.1718750, 34448.7539062

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6259932, upper bound: 19341.9917611
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18657.2773438, 13784.1298828, -15385.0976562, 11382.8056641, -30040.0820312, 29169.2246094
1: -15102.3857422, 13346.4589844, -12452.2431641, 11018.8076172, -26121.1933594, 25798.6953125
2: -22039.4648438, 14576.5332031, -18158.9550781, 12037.1357422, -34076.6015625, 32735.4882812
3: -7609.1791992, 20777.9628906, -6284.6875000, 17098.0156250, -24707.1953125, 27062.6503906
4: -24205.4062500, 14394.3369141, -19956.5410156, 11899.9658203, -36105.3710938, 34350.8750000

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
time: 0.97 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18657.2773438, 13784.1298828, -16127.6875000, 11970.2519531, -30627.5273438, 29911.8164062
1: -15102.3857422, 13346.4589844, -13054.4599609, 11591.3388672, -26693.7226562, 26400.9160156
2: -22039.4648438, 14576.5332031, -19041.2832031, 12659.5800781, -34699.0468750, 33617.8164062
3: -7609.1791992, 20777.9628906, -6605.4584961, 17970.6660156, -25579.8457031, 27383.4218750
4: -24205.4062500, 14394.3369141, -20919.1621094, 12504.9296875, -36710.3359375, 35313.4921875

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18657.2773438, 13784.1298828, -17528.5878906, 12883.6152344, -31540.8925781, 31312.7187500
1: -15102.3857422, 13346.4589844, -14189.0595703, 12475.7558594, -27578.1367188, 27535.5156250
2: -22039.4648438, 14576.5332031, -20702.6328125, 13623.4335938, -35662.8984375, 35279.1640625
3: -7609.1791992, 20777.9628906, -7127.1171875, 19470.5488281, -27079.7285156, 27905.0781250
4: -24205.4062500, 14394.3369141, -22745.2597656, 13462.9326172, -37668.3398438, 37139.5937500

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18561.1269531, 13712.1699219, -19088.4375000, 13986.9843750, -32548.1093750, 32800.6093750
1: -15024.8369141, 13276.6875000, -15449.7695312, 13549.2099609, -28574.0449219, 28726.4570312
2: -21926.0195312, 14499.8437500, -22523.0351562, 14793.2324219, -36719.2460938, 37022.8789062
3: -7568.5371094, 20670.4023438, -7728.8330078, 21184.5917969, -28753.1289062, 28399.2343750
4: -24080.2246094, 14318.3662109, -24736.4765625, 14612.7480469, -38692.9726562, 39054.8281250

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
time: 1.18 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
time: 1.21 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.65 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418450
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418450
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871412
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9871575, upper bound: 19342.9871412
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.2522304, upper bound: 19342.7864284
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.65
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.6841296
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.2522304, upper bound: 19343.0269200
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.65
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.8680237
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8593958
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.8593958
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9528033
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.5170112, upper bound: 19342.9528033
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.6259932, upper bound: 19341.9917611
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.6259932, upper bound: 19341.9917611
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.8563594, upper bound: 19342.5170112
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.65
Output dim: 3, lower bound: -19342.8048950, upper bound: 19342.8002783

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14714.6064453, 10904.2587891, -15379.9941406, 11379.1894531, -26093.7929688, 26284.2539062
1: -11909.0263672, 10553.8779297, -12448.1044922, 11015.2929688, -22924.3203125, 23001.9765625
2: -17367.7363281, 11530.3730469, -18152.9394531, 12033.3173828, -29401.0507812, 29683.3125000
3: -6016.9643555, 16355.0917969, -6282.6357422, 17092.3789062, -23109.3437500, 22637.7265625
4: -19086.1601562, 11398.5839844, -19949.9121094, 11896.1787109, -30982.3339844, 31348.4941406

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3223565, upper bound: 19340.5586628
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.3746912, upper bound: 19340.1142489
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15291.7265625, 11311.9248047, -15385.0976562, 11382.8056641, -26674.5312500, 26697.0214844
1: -12376.9931641, 10950.3818359, -12452.2431641, 11018.8076172, -23395.8007812, 23402.6210938
2: -18049.3437500, 11962.5625000, -18158.9550781, 12037.1357422, -30086.4804688, 30121.5156250
3: -6245.9423828, 16994.0996094, -6284.6875000, 17098.0156250, -23343.9570312, 23278.7851562
4: -19835.9296875, 11825.4414062, -19956.5410156, 11899.9658203, -31735.8945312, 31781.9824219

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3924719, upper bound: 19340.8788743
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.4078459, upper bound: 19340.4078459
time: 2.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14714.6064453, 10904.2587891, -16122.6093750, 11966.5927734, -26681.1972656, 27026.8671875
1: -11909.0263672, 10553.8779297, -13050.3476562, 11587.7851562, -23496.8125000, 23604.2207031
2: -17367.7363281, 11530.3730469, -19035.2929688, 12655.7167969, -30023.4531250, 30565.6660156
3: -6016.9643555, 16355.0917969, -6603.4130859, 17965.0332031, -23981.9980469, 22958.5039062
4: -19086.1601562, 11398.5839844, -20912.5585938, 12501.1044922, -31587.2636719, 32311.1406250

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2681078, upper bound: 19340.5568840
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15291.7265625, 11311.9248047, -16127.6875000, 11970.2519531, -27261.9765625, 27439.6132812
1: -12376.9931641, 10950.3818359, -13054.4599609, 11591.3388672, -23968.3320312, 24004.8417969
2: -18049.3437500, 11962.5625000, -19041.2832031, 12659.5800781, -30708.9238281, 31003.8417969
3: -6245.9423828, 16994.0996094, -6605.4584961, 17970.6660156, -24216.6093750, 23599.5585938
4: -19835.9296875, 11825.4414062, -20919.1621094, 12504.9296875, -32340.8593750, 32744.6035156

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.3421803, upper bound: 19340.8770789
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -15379.9941406, 11379.1894531, -26861.5937500, 26887.3476562
1: -12532.0966797, 11141.1728516, -12448.1044922, 11015.2929688, -23547.3906250, 23589.2734375
2: -18279.1914062, 12169.2402344, -18152.9394531, 12033.3173828, -30312.5058594, 30322.1796875
3: -6347.8457031, 17254.7207031, -6282.6357422, 17092.3789062, -23440.2246094, 23537.3554688
4: -20080.5488281, 12020.0205078, -19949.9121094, 11896.1787109, -31976.7226562, 31969.9335938

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -15385.0976562, 11382.8056641, -27417.7949219, 27286.5859375
1: -12979.6025391, 11524.6894531, -12452.2431641, 11018.8076172, -23998.4101562, 23976.9257812
2: -18932.5468750, 12586.2998047, -18158.9550781, 12037.1357422, -30969.6835938, 30745.2539062
3: -6566.6435547, 17867.9726562, -6284.6875000, 17098.0156250, -23664.6562500, 24152.6582031
4: -20799.5703125, 12432.6601562, -19956.5410156, 11899.9658203, -32699.5351562, 32389.2011719

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -16122.6093750, 11966.5927734, -27448.9960938, 27629.9628906
1: -12532.0966797, 11141.1728516, -13050.3476562, 11587.7851562, -24119.8828125, 24191.5175781
2: -18279.1914062, 12169.2402344, -19035.2929688, 12655.7167969, -30934.9082031, 31204.5332031
3: -6347.8457031, 17254.7207031, -6603.4130859, 17965.0332031, -24312.8789062, 23858.1328125
4: -20080.5488281, 12020.0205078, -20912.5585938, 12501.1044922, -32581.6523438, 32932.5742188

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -16127.6875000, 11970.2519531, -28005.2402344, 28029.1796875
1: -12979.6025391, 11524.6894531, -13054.4599609, 11591.3388672, -24570.9414062, 24579.1484375
2: -18932.5468750, 12586.2998047, -19041.2832031, 12659.5800781, -31592.1250000, 31627.5820312
3: -6566.6435547, 17867.9726562, -6605.4584961, 17970.6660156, -24537.3085938, 24473.4316406
4: -20799.5703125, 12432.6601562, -20919.1621094, 12504.9296875, -33304.5000000, 33351.8203125

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15032.3144531, 11120.3095703, -17610.9257812, 12945.5888672, -27977.9023438, 28731.2343750
1: -12167.4804688, 10765.2890625, -14255.5488281, 12535.5800781, -24703.0605469, 25020.8359375
2: -17744.0605469, 11760.3574219, -20799.7832031, 13690.3691406, -31434.4296875, 32560.1386719
3: -6139.8403320, 16706.5000000, -7160.5898438, 19561.6269531, -25701.4648438, 23867.0878906
4: -19501.0253906, 11626.1728516, -22852.2402344, 13529.6064453, -33030.6289062, 34478.4140625

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.6841296
time: 4.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.6841296
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15032.3144531, 11120.3095703, -18657.2773438, 13784.1298828, -28816.4433594, 29777.5859375
1: -12167.4804688, 10765.2890625, -15102.3857422, 13346.4589844, -25513.9375000, 25867.6699219
2: -17744.0605469, 11760.3574219, -22039.4648438, 14576.5332031, -32320.5937500, 33799.8203125
3: -6139.8403320, 16706.5000000, -7609.1791992, 20777.9628906, -26917.8027344, 24315.6796875
4: -19501.0253906, 11626.1728516, -24205.4062500, 14394.3369141, -33895.3632812, 35831.5781250

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2248005, upper bound: 19340.8680237
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2248005, upper bound: 19340.8680237
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -17605.9707031, 12942.0947266, -28424.5000000, 29113.3222656
1: -12532.0966797, 11141.1728516, -14251.5302734, 12532.1826172, -25064.2792969, 25392.7031250
2: -18279.1914062, 12169.2402344, -20793.9414062, 13686.6777344, -31965.8671875, 32963.1796875
3: -6347.8457031, 17254.7207031, -7158.6088867, 19556.1640625, -25904.0078125, 24413.3300781
4: -20080.5488281, 12020.0205078, -22845.8007812, 13525.9453125, -33606.4921875, 34865.8203125

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.6259932
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.8593958
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -17610.9257812, 12945.5888672, -28980.5781250, 29512.4160156
1: -12979.6025391, 11524.6894531, -14255.5488281, 12535.5800781, -25515.1796875, 25780.2343750
2: -18932.5468750, 12586.2998047, -20799.7832031, 13690.3691406, -32622.9160156, 33386.0781250
3: -6566.6435547, 17867.9726562, -7160.5898438, 19561.6269531, -26128.2656250, 25028.5585938
4: -20799.5703125, 12432.6601562, -22852.2402344, 13529.6064453, -34329.1718750, 35284.8984375

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.6259932
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.8593958
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -18652.4218750, 13780.6884766, -29263.0937500, 30159.7753906
1: -12532.0966797, 11141.1728516, -15098.4580078, 13343.0937500, -25875.1894531, 26239.6289062
2: -18279.1914062, 12169.2402344, -22033.7480469, 14572.9111328, -32852.1015625, 34202.9882812
3: -6347.8457031, 17254.7207031, -7607.2363281, 20772.6113281, -27120.4531250, 24861.9570312
4: -20080.5488281, 12020.0205078, -24199.0957031, 14390.7294922, -34471.2773438, 36219.1171875

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9113640
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9528033
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -18657.2773438, 13784.1298828, -29819.1191406, 30558.7695312
1: -12979.6025391, 11524.6894531, -15102.3857422, 13346.4589844, -26326.0566406, 26627.0683594
2: -18932.5468750, 12586.2998047, -22039.4648438, 14576.5332031, -33509.0781250, 34625.7656250
3: -6566.6435547, 17867.9726562, -7609.1791992, 20777.9628906, -27344.6054688, 25477.1523438
4: -20799.5703125, 12432.6601562, -24205.4062500, 14394.3369141, -35193.9062500, 36638.0664062

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9113640
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9528033
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17505.8066406, 12864.6865234, -15385.0976562, 11382.8056641, -28888.6113281, 28249.7832031
1: -14170.8173828, 12457.5332031, -12452.2431641, 11018.8076172, -25189.6250000, 24909.7695312
2: -20676.2421875, 13605.6816406, -18158.9550781, 12037.1357422, -32713.3789062, 31764.6367188
3: -7116.5336914, 19443.4218750, -6284.6875000, 17098.0156250, -24214.5468750, 25728.1093750
4: -22716.3847656, 13445.1572266, -19956.5410156, 11899.9658203, -34616.3476562, 33401.6953125

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7864284, upper bound: 19342.2522304
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6841296, upper bound: 19341.6951246
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -17505.8066406, 12864.6865234, -16127.6875000, 11970.2519531, -29476.0566406, 28992.3750000
1: -14170.8173828, 12457.5332031, -13054.4599609, 11591.3388672, -25762.1562500, 25511.9902344
2: -20676.2421875, 13605.6816406, -19041.2832031, 12659.5800781, -33335.8125000, 32646.9648438
3: -7116.5336914, 19443.4218750, -6605.4584961, 17970.6660156, -25087.1992188, 26048.8808594
4: -22716.3847656, 13445.1572266, -20919.1621094, 12504.9296875, -35221.3125000, 34364.3203125

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7389709, upper bound: 19342.2507117
time: 0.92 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8593958, upper bound: 19342.5170112
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8593958, upper bound: 19342.5170112
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18033.1074219, 13335.7832031, -15379.9941406, 11379.1894531, -29412.2949219, 28715.7773438
1: -14597.0009766, 12910.7148438, -12448.1044922, 11015.2929688, -25612.2929688, 25358.8164062
2: -21301.1367188, 14102.8271484, -18152.9394531, 12033.3173828, -33334.4531250, 32255.7656250
3: -7358.1684570, 20083.9648438, -6282.6357422, 17092.3789062, -24450.5468750, 26366.6015625
4: -23392.2519531, 13924.3330078, -19949.9121094, 11896.1787109, -35288.4296875, 33874.2460938

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0269063, upper bound: 19342.8169221
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.8677600, upper bound: 19342.2233612
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -18548.9160156, 13700.9375000, -15385.0976562, 11382.8056641, -29931.7226562, 29086.0332031
1: -15014.7841797, 13267.2753906, -12452.2431641, 11018.8076172, -26033.5917969, 25719.5156250
2: -21912.0761719, 14489.8896484, -18158.9550781, 12037.1357422, -33949.2109375, 32648.8437500
3: -7563.5458984, 20656.7304688, -6284.6875000, 17098.0156250, -24661.5625000, 26941.4140625
4: -24065.4082031, 14307.9609375, -19956.5410156, 11899.9658203, -35965.3750000, 34264.4921875

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0269063, upper bound: 19342.8192480
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.8680237, upper bound: 19342.2246841
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18033.1074219, 13335.7832031, -16122.6093750, 11966.5927734, -29999.6992188, 29458.3925781
1: -14597.0009766, 12910.7148438, -13050.3476562, 11587.7851562, -26184.7851562, 25961.0625000
2: -21301.1367188, 14102.8271484, -19035.2929688, 12655.7167969, -33956.8515625, 33138.1171875
3: -7358.1684570, 20083.9648438, -6603.4130859, 17965.0332031, -25323.2011719, 26687.3789062
4: -23392.2519531, 13924.3330078, -20912.5585938, 12501.1044922, -35893.3554688, 34836.8906250

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9721882, upper bound: 19342.8154586
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
time: 0.91 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18548.9160156, 13700.9375000, -16127.6875000, 11970.2519531, -30519.1660156, 29828.6250000
1: -15014.7841797, 13267.2753906, -13054.4599609, 11591.3388672, -26606.1230469, 26321.7343750
2: -21912.0761719, 14489.8896484, -19041.2832031, 12659.5800781, -34571.6484375, 33531.1718750
3: -7563.5458984, 20656.7304688, -6605.4584961, 17970.6660156, -25534.2109375, 27262.1894531
4: -24065.4082031, 14307.9609375, -20919.1621094, 12504.9296875, -36570.3359375, 35227.1132812

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9721882, upper bound: 19342.8177835
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -17528.5878906, 12883.6152344, -31309.6660156, 31142.4062500
1: -14914.9316406, 13181.6396484, -14189.0595703, 12475.7558594, -27390.6855469, 27370.6933594
2: -21765.7910156, 14396.5722656, -20702.6328125, 13623.4335938, -35389.2226562, 35099.2031250
3: -7514.8212891, 20520.4863281, -7127.1171875, 19470.5488281, -26985.3671875, 27647.6035156
4: -23905.1875000, 14216.1181641, -22745.2597656, 13462.9326172, -37368.1210938, 36961.3710938

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.7028136
time: 1.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8093057
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -17528.5878906, 12883.6152344, -32934.0234375, 32283.4687500
1: -16229.7929688, 14288.9218750, -14189.0595703, 12475.7558594, -28705.5488281, 28477.9804688
2: -23660.3769531, 15599.0976562, -20702.6328125, 13623.4335938, -37283.8125000, 36301.7304688
3: -8136.4487305, 22301.8632812, -7127.1171875, 19470.5488281, -27606.9980469, 29428.9785156
4: -25976.0585938, 15398.0332031, -22745.2597656, 13462.9326172, -39438.9921875, 38143.2929688

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.7028136
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8093057
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -19088.4375000, 13986.9843750, -32413.0351562, 32702.2558594
1: -14914.9316406, 13181.6396484, -15449.7695312, 13549.2099609, -28464.1347656, 28631.4082031
2: -21765.7910156, 14396.5722656, -22523.0351562, 14793.2324219, -36559.0195312, 36919.6054688
3: -7514.8212891, 20520.4863281, -7728.8330078, 21184.5917969, -28699.4140625, 28249.3203125
4: -23905.1875000, 14216.1181641, -24736.4765625, 14612.7480469, -38517.9375000, 38952.5820312

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.5777940
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7976103
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -19088.4375000, 13986.9843750, -34037.3906250, 33843.3203125
1: -16229.7929688, 14288.9218750, -15449.7695312, 13549.2099609, -29779.0039062, 29738.6914062
2: -23660.3769531, 15599.0976562, -22523.0351562, 14793.2324219, -38453.6093750, 38122.1328125
3: -8136.4487305, 22301.8632812, -7728.8330078, 21184.5917969, -29321.0410156, 30030.6953125
4: -25976.0585938, 15398.0332031, -24736.4765625, 14612.7480469, -40588.8046875, 40134.5000000

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.5777940
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7976103
time: 0.99 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.96 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.3223565, upper bound: 19340.5586628
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19340.3746912, upper bound: 19340.1142489
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.3924719, upper bound: 19340.8788743
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19340.4078459, upper bound: 19340.4078459
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9884184, upper bound: 19343.0418366
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19343.0418366, upper bound: 19342.9884184
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9871412, upper bound: 19342.9871412
IS_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.6841296
IS_A1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19341.6952342, upper bound: 19340.6841296
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.2248005, upper bound: 19340.8680237
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.2248005, upper bound: 19340.8680237
IS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.6259932
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.8593958
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.6259932
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.8593958
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9113640
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9528033
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9113640
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.8823250, upper bound: 19342.9528033
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.7864284, upper bound: 19342.2522304
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19340.6841296, upper bound: 19341.6951246
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.8593958, upper bound: 19342.5170112
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.8593958, upper bound: 19342.5170112
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19343.0269063, upper bound: 19342.8169221
IS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19340.8677600, upper bound: 19342.2233612
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19343.0269063, upper bound: 19342.8192480
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19340.8680237, upper bound: 19342.2246841
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8889351
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.9778274, upper bound: 19342.8920905
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.7028136
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8093057
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.7028136
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8093057
IS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.5777940
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7976103
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.5777940
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.96
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7976103

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14714.6064453, 10904.2587891, -15482.4042969, 11507.3535156, -26221.9570312, 26386.6640625
1: -11909.0263672, 10553.8779297, -12532.0966797, 11141.1728516, -23050.1992188, 23085.9726562
2: -17367.7363281, 11530.3730469, -18279.1914062, 12169.2402344, -29536.9765625, 29809.5625000
3: -6016.9643555, 16355.0917969, -6347.8457031, 17254.7207031, -23271.6855469, 22702.9355469
4: -19086.1601562, 11398.5839844, -20080.5488281, 12020.0205078, -31106.1796875, 31479.1328125

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335634
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14714.6064453, 10904.2587891, -16034.9892578, 11901.4921875, -26616.0937500, 26939.2480469
1: -11909.0263672, 10553.8779297, -12979.6025391, 11524.6894531, -23433.7128906, 23533.4765625
2: -17367.7363281, 11530.3730469, -18932.5468750, 12586.2998047, -29954.0332031, 30462.9199219
3: -6016.9643555, 16355.0917969, -6566.6435547, 17867.9726562, -23884.9375000, 22921.7304688
4: -19086.1601562, 11398.5839844, -20799.5703125, 12432.6601562, -31518.8203125, 32198.1523438

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335634
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15291.7265625, 11311.9248047, -15482.4042969, 11507.3535156, -26799.0800781, 26794.3281250
1: -12376.9931641, 10950.3818359, -12532.0966797, 11141.1728516, -23518.1621094, 23482.4785156
2: -18049.3437500, 11962.5625000, -18279.1914062, 12169.2402344, -30218.5839844, 30241.7519531
3: -6245.9423828, 16994.0996094, -6347.8457031, 17254.7207031, -23500.6640625, 23341.9433594
4: -19835.9296875, 11825.4414062, -20080.5488281, 12020.0205078, -31855.9492188, 31905.9902344

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335598
time: 8.04 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9758356, upper bound: 19343.0311808
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15291.7265625, 11311.9248047, -16034.9892578, 11901.4921875, -27193.2187500, 27346.9140625
1: -12376.9931641, 10950.3818359, -12979.6025391, 11524.6894531, -23901.6777344, 23929.9824219
2: -18049.3437500, 11962.5625000, -18932.5468750, 12586.2998047, -30635.6445312, 30895.1074219
3: -6245.9423828, 16994.0996094, -6566.6435547, 17867.9726562, -24113.9140625, 23560.7402344
4: -19835.9296875, 11825.4414062, -20799.5703125, 12432.6601562, -32268.5898438, 32625.0117188

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335598
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9758356, upper bound: 19343.0311808
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -14714.6064453, 10904.2587891, -26386.6640625, 26221.9550781
1: -12532.0966797, 11141.1728516, -11909.0263672, 10553.8779297, -23085.9726562, 23050.1992188
2: -18279.1914062, 12169.2402344, -17367.7363281, 11530.3730469, -29809.5644531, 29536.9765625
3: -6347.8457031, 17254.7207031, -6016.9643555, 16355.0917969, -22702.9355469, 23271.6855469
4: -20080.5488281, 12020.0205078, -19086.1601562, 11398.5839844, -31479.1308594, 31106.1796875

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.5568840, upper bound: 19342.2681078
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0379924, upper bound: 19342.9835734
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0326363, upper bound: 19342.9798127
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -15291.7265625, 11311.9248047, -26794.3281250, 26799.0800781
1: -12532.0966797, 11141.1728516, -12376.9931641, 10950.3818359, -23482.4785156, 23518.1640625
2: -18279.1914062, 12169.2402344, -18049.3437500, 11962.5625000, -30241.7519531, 30218.5839844
3: -6347.8457031, 17254.7207031, -6245.9423828, 16994.0996094, -23341.9433594, 23500.6640625
4: -20080.5488281, 12020.0205078, -19835.9296875, 11825.4414062, -31905.9902344, 31855.9492188

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.5568840, upper bound: 19342.3421803
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0379924, upper bound: 19342.9835734
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0326363, upper bound: 19342.9798127
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -14714.6064453, 10904.2587891, -26939.2480469, 26616.0937500
1: -12979.6025391, 11524.6894531, -11909.0263672, 10553.8779297, -23533.4765625, 23433.7148438
2: -18932.5468750, 12586.2998047, -17367.7363281, 11530.3730469, -30462.9199219, 29954.0332031
3: -6566.6435547, 17867.9726562, -6016.9643555, 16355.0917969, -22921.7304688, 23884.9375000
4: -20799.5703125, 12432.6601562, -19086.1601562, 11398.5839844, -32198.1523438, 31518.8203125

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.5568840, upper bound: 19342.2681078
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0379514, upper bound: 19342.9835734
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0311808, upper bound: 19342.9758356
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -15291.7265625, 11311.9248047, -27346.9140625, 27193.2187500
1: -12979.6025391, 11524.6894531, -12376.9931641, 10950.3818359, -23929.9824219, 23901.6796875
2: -18932.5468750, 12586.2998047, -18049.3437500, 11962.5625000, -30895.1074219, 30635.6445312
3: -6566.6435547, 17867.9726562, -6245.9423828, 16994.0996094, -23560.7402344, 24113.9140625
4: -20799.5703125, 12432.6601562, -19835.9296875, 11825.4414062, -32625.0117188, 32268.5898438

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.5568840, upper bound: 19342.3276955
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0379514, upper bound: 19342.9835734
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0311808, upper bound: 19342.9758356
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -15482.4042969, 11507.3535156, -26989.7578125, 26989.7578125
1: -12532.0966797, 11141.1728516, -12532.0966797, 11141.1728516, -23673.2675781, 23673.2675781
2: -18279.1914062, 12169.2402344, -18279.1914062, 12169.2402344, -30448.4316406, 30448.4316406
3: -6347.8457031, 17254.7207031, -6347.8457031, 17254.7207031, -23602.5664062, 23602.5644531
4: -20080.5488281, 12020.0205078, -20080.5488281, 12020.0205078, -32100.5703125, 32100.5683594

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9790818, upper bound: 19342.9800046
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9737223, upper bound: 19342.9762439
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -16034.9892578, 11901.4921875, -27383.8964844, 27542.3437500
1: -12532.0966797, 11141.1728516, -12979.6025391, 11524.6894531, -24056.7812500, 24120.7734375
2: -18279.1914062, 12169.2402344, -18932.5468750, 12586.2998047, -30865.4921875, 31101.7871094
3: -6347.8457031, 17254.7207031, -6566.6435547, 17867.9726562, -24215.8183594, 23821.3632812
4: -20080.5488281, 12020.0205078, -20799.5703125, 12432.6601562, -32513.2089844, 32819.5898438

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9790818, upper bound: 19342.9800046
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9737223, upper bound: 19342.9762439
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -15482.4042969, 11507.3535156, -27542.3437500, 27383.8964844
1: -12979.6025391, 11524.6894531, -12532.0966797, 11141.1728516, -24120.7714844, 24056.7832031
2: -18932.5468750, 12586.2998047, -18279.1914062, 12169.2402344, -31101.7871094, 30865.4921875
3: -6566.6435547, 17867.9726562, -6347.8457031, 17254.7207031, -23821.3632812, 24215.8164062
4: -20799.5703125, 12432.6601562, -20080.5488281, 12020.0205078, -32819.5898438, 32513.2089844

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9790414, upper bound: 19342.9800046
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9722668, upper bound: 19342.9722668
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -16034.9892578, 11901.4921875, -27936.4804688, 27936.4804688
1: -12979.6025391, 11524.6894531, -12979.6025391, 11524.6894531, -24504.2851562, 24504.2871094
2: -18932.5468750, 12586.2998047, -18932.5468750, 12586.2998047, -31518.8476562, 31518.8457031
3: -6566.6435547, 17867.9726562, -6566.6435547, 17867.9726562, -24434.6132812, 24434.6132812
4: -20799.5703125, 12432.6601562, -20799.5703125, 12432.6601562, -33232.2304688, 33232.2304688

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9790414, upper bound: 19342.9800046
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9722668, upper bound: 19342.9722668
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -17505.8066406, 12864.6865234, -28347.0898438, 29013.1582031
1: -12532.0966797, 11141.1728516, -14170.8173828, 12457.5332031, -24989.6269531, 25311.9902344
2: -18279.1914062, 12169.2402344, -20676.2421875, 13605.6816406, -31884.8730469, 32845.4843750
3: -6347.8457031, 17254.7207031, -7116.5336914, 19443.4218750, -25791.2675781, 24371.2519531
4: -20080.5488281, 12020.0205078, -22716.3847656, 13445.1572266, -33525.7070312, 34736.4062500

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.8111590, upper bound: 19342.7389709
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.9920384, upper bound: 19342.7181527
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.9863915, upper bound: 19342.7143948
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -17505.8066406, 12864.6865234, -28899.6757812, 29407.2988281
1: -12979.6025391, 11524.6894531, -14170.8173828, 12457.5332031, -25437.1328125, 25695.5058594
2: -18932.5468750, 12586.2998047, -20676.2421875, 13605.6816406, -32538.2285156, 33262.5312500
3: -6566.6435547, 17867.9726562, -7116.5336914, 19443.4218750, -26010.0664062, 24984.5039062
4: -20799.5703125, 12432.6601562, -22716.3847656, 13445.1572266, -34244.7265625, 35149.0468750

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.8111590, upper bound: 19342.6082979
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.6587280
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.9850191, upper bound: 19342.6192967
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -18033.1074219, 13335.7832031, -28818.1875000, 29540.4570312
1: -12532.0966797, 11141.1728516, -14597.0009766, 12910.7148438, -25442.8125000, 25738.1738281
2: -18279.1914062, 12169.2402344, -21301.1367188, 14102.8271484, -32382.0195312, 33470.3750000
3: -6347.8457031, 17254.7207031, -7358.1684570, 20083.9648438, -26431.8066406, 24612.8886719
4: -20080.5488281, 12020.0205078, -23392.2519531, 13924.3330078, -34004.8828125, 35412.2734375

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8105661, upper bound: 19342.8627790
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8602283, upper bound: 19342.9113639
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8539360, upper bound: 19342.8802317
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -18548.9160156, 13700.9375000, -29183.3417969, 30056.2695312
1: -12532.0966797, 11141.1728516, -15014.7841797, 13267.2753906, -25799.3710938, 26155.9531250
2: -18279.1914062, 12169.2402344, -21912.0761719, 14489.8896484, -32769.0820312, 34081.3164062
3: -6347.8457031, 17254.7207031, -7563.5458984, 20656.7304688, -27004.5722656, 24818.2656250
4: -20080.5488281, 12020.0205078, -24065.4082031, 14307.9609375, -34388.5078125, 36085.4296875

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8105661, upper bound: 19342.9277785
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8602283, upper bound: 19342.9278552
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8539360, upper bound: 19342.9033768
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -18033.1074219, 13335.7832031, -29370.7734375, 29934.5957031
1: -12979.6025391, 11524.6894531, -14597.0009766, 12910.7148438, -25890.3164062, 26121.6914062
2: -18932.5468750, 12586.2998047, -21301.1367188, 14102.8271484, -33035.3750000, 33887.4335938
3: -6566.6435547, 17867.9726562, -7358.1684570, 20083.9648438, -26650.6074219, 25226.1386719
4: -20799.5703125, 12432.6601562, -23392.2519531, 13924.3330078, -34723.9023438, 35824.9140625

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8102478, upper bound: 19342.8627790
time: 1.10 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8599095, upper bound: 19342.9113639
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8514304, upper bound: 19342.8714795
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -18548.9160156, 13700.9375000, -29735.9257812, 30450.4062500
1: -12979.6025391, 11524.6894531, -15014.7841797, 13267.2753906, -26246.8769531, 26539.4687500
2: -18932.5468750, 12586.2998047, -21912.0761719, 14489.8896484, -33422.4335938, 34498.3710938
3: -6566.6435547, 17867.9726562, -7563.5458984, 20656.7304688, -27223.3671875, 25431.5195312
4: -20799.5703125, 12432.6601562, -24065.4082031, 14307.9609375, -35107.5273438, 36498.0703125

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8102478, upper bound: 19342.8673898
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8599095, upper bound: 19342.9114108
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8514304, upper bound: 19342.8714794
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -17505.8066406, 12864.6865234, -15032.3144531, 11120.3095703, -28626.1132812, 27897.0000000
1: -14170.8173828, 12457.5332031, -12167.4804688, 10765.2890625, -24936.1054688, 24625.0117188
2: -20676.2421875, 13605.6816406, -17744.0605469, 11760.3574219, -32436.5996094, 31349.7421875
3: -7116.5336914, 19443.4218750, -6139.8403320, 16706.5000000, -23823.0292969, 25583.2617188
4: -22716.3847656, 13445.1572266, -19501.0253906, 11626.1728516, -34342.5585938, 32946.1796875

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6841296, upper bound: 19341.6951246
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6841296, upper bound: 19341.6951246
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -17505.8066406, 12864.6865234, -15482.4042969, 11507.3535156, -29013.1601562, 28347.0898438
1: -14170.8173828, 12457.5332031, -12532.0966797, 11141.1728516, -25311.9902344, 24989.6269531
2: -20676.2421875, 13605.6816406, -18279.1914062, 12169.2402344, -32845.4843750, 31884.8730469
3: -7116.5336914, 19443.4218750, -6347.8457031, 17254.7207031, -24371.2519531, 25791.2675781
4: -22716.3847656, 13445.1572266, -20080.5488281, 12020.0205078, -34736.4062500, 33525.7070312

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8220282, upper bound: 19342.4986184
time: 0.99 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7104673, upper bound: 19342.2931491
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -17505.8066406, 12864.6865234, -16034.9892578, 11901.4921875, -29407.2988281, 28899.6757812
1: -14170.8173828, 12457.5332031, -12979.6025391, 11524.6894531, -25695.5039062, 25437.1328125
2: -20676.2421875, 13605.6816406, -18932.5468750, 12586.2998047, -33262.5351562, 32538.2265625
3: -7116.5336914, 19443.4218750, -6566.6435547, 17867.9726562, -24984.5039062, 26010.0664062
4: -22716.3847656, 13445.1572266, -20799.5703125, 12432.6601562, -35149.0468750, 34244.7265625

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8220282, upper bound: 19342.4986184
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7104673, upper bound: 19342.2931491
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18033.1074219, 13335.7832031, -15027.1240234, 11116.6191406, -29149.7207031, 28362.9023438
1: -14597.0009766, 12910.7148438, -12163.2705078, 10761.7031250, -25358.7031250, 25073.9843750
2: -21301.1367188, 14102.8271484, -17737.9394531, 11756.4589844, -33057.5937500, 31840.7656250
3: -7358.1684570, 20083.9648438, -6137.7504883, 16700.7558594, -24058.9238281, 26221.7109375
4: -23392.2519531, 13924.3330078, -19494.2792969, 11622.3095703, -35014.5625000, 33418.6132812

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.8677600, upper bound: 19342.2233612
time: 1.01 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.8677600, upper bound: 19342.2233612
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18548.9160156, 13700.9375000, -15032.3144531, 11120.3095703, -29669.2226562, 28733.2519531
1: -15014.7841797, 13267.2753906, -12167.4804688, 10765.2890625, -25780.0722656, 25434.7558594
2: -21912.0761719, 14489.8896484, -17744.0605469, 11760.3574219, -33672.4335938, 32233.9492188
3: -7563.5458984, 20656.7304688, -6139.8403320, 16706.5000000, -24270.0449219, 26796.5683594
4: -24065.4082031, 14307.9609375, -19501.0253906, 11626.1728516, -35691.5820312, 33808.9804688

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.8680237, upper bound: 19342.2246841
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.8680237, upper bound: 19342.2246841
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -18033.1074219, 13335.7832031, -15482.4042969, 11507.3535156, -29540.4589844, 28818.1875000
1: -14597.0009766, 12910.7148438, -12532.0966797, 11141.1728516, -25738.1738281, 25442.8125000
2: -21301.1367188, 14102.8271484, -18279.1914062, 12169.2402344, -33470.3750000, 32382.0195312
3: -7358.1684570, 20083.9648438, -6347.8457031, 17254.7207031, -24612.8886719, 26431.8085938
4: -23392.2519531, 13924.3330078, -20080.5488281, 12020.0205078, -35412.2734375, 34004.8828125

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8819983
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9628283, upper bound: 19342.8598215
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -18033.1074219, 13335.7832031, -16034.9892578, 11901.4921875, -29934.5957031, 29370.7734375
1: -14597.0009766, 12910.7148438, -12979.6025391, 11524.6894531, -26121.6914062, 25890.3164062
2: -21301.1367188, 14102.8271484, -18932.5468750, 12586.2998047, -33887.4335938, 33035.3750000
3: -7358.1684570, 20083.9648438, -6566.6435547, 17867.9726562, -25226.1406250, 26650.6054688
4: -23392.2519531, 13924.3330078, -20799.5703125, 12432.6601562, -35824.9140625, 34723.9023438

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8819983
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9628283, upper bound: 19342.8598215
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18548.9160156, 13700.9375000, -15482.4042969, 11507.3535156, -30056.2695312, 29183.3417969
1: -15014.7841797, 13267.2753906, -12532.0966797, 11141.1728516, -26155.9550781, 25799.3710938
2: -21912.0761719, 14489.8896484, -18279.1914062, 12169.2402344, -34081.3164062, 32769.0820312
3: -7563.5458984, 20656.7304688, -6347.8457031, 17254.7207031, -24818.2656250, 27004.5722656
4: -24065.4082031, 14307.9609375, -20080.5488281, 12020.0205078, -36085.4296875, 34388.5078125

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8851536
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9626911, upper bound: 19342.8598215
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18548.9160156, 13700.9375000, -16034.9892578, 11901.4921875, -30450.4062500, 29735.9257812
1: -15014.7841797, 13267.2753906, -12979.6025391, 11524.6894531, -26539.4687500, 26246.8750000
2: -21912.0761719, 14489.8896484, -18932.5468750, 12586.2998047, -34498.3710938, 33422.4375000
3: -7563.5458984, 20656.7304688, -6566.6435547, 17867.9726562, -25431.5195312, 27223.3691406
4: -24065.4082031, 14307.9609375, -20799.5703125, 12432.6601562, -36498.0703125, 35107.5273438

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8851536
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9626911, upper bound: 19342.8598215
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -17343.9531250, 12745.0712891, -31171.1210938, 30957.7714844
1: -14914.9316406, 13181.6396484, -14039.0566406, 12342.1806641, -27257.1093750, 27220.6894531
2: -21765.7910156, 14396.5722656, -20484.5214844, 13477.2841797, -35243.0742188, 34881.0859375
3: -7514.8212891, 20520.4863281, -7052.6738281, 19265.0742188, -26779.8906250, 27573.1601562
4: -23905.1875000, 14216.1181641, -22506.4316406, 13317.8515625, -37223.0390625, 36722.5507812

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5067550, upper bound: 19342.7803949
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5067610, upper bound: 19342.7840574
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -18426.0507812, 13613.8183594, -32039.8691406, 32039.8691406
1: -14914.9316406, 13181.6396484, -14914.9316406, 13181.6396484, -28096.5625000, 28096.5625000
2: -21765.7910156, 14396.5722656, -21765.7910156, 14396.5722656, -36162.3593750, 36162.3593750
3: -7514.8212891, 20520.4863281, -7514.8212891, 20520.4863281, -28035.3085938, 28035.3066406
4: -23905.1875000, 14216.1181641, -23905.1875000, 14216.1181641, -38121.3046875, 38121.3046875

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5067550, upper bound: 19342.8784846
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5067610, upper bound: 19342.8813183
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -17343.9531250, 12745.0712891, -32795.4804688, 32098.8359375
1: -16229.7929688, 14288.9218750, -14039.0566406, 12342.1806641, -28571.9726562, 28327.9765625
2: -23660.3769531, 15599.0976562, -20484.5214844, 13477.2841797, -37137.6601562, 36083.6171875
3: -8136.4487305, 22301.8632812, -7052.6738281, 19265.0742188, -27401.5234375, 29354.5371094
4: -25976.0585938, 15398.0332031, -22506.4316406, 13317.8515625, -39293.9101562, 37904.4648438

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.6944202
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.7028136
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -18426.0507812, 13613.8183594, -33664.2265625, 33180.9296875
1: -16229.7929688, 14288.9218750, -14914.9316406, 13181.6396484, -29411.4335938, 29203.8515625
2: -23660.3769531, 15599.0976562, -21765.7910156, 14396.5722656, -38056.9492188, 37364.8867188
3: -8136.4487305, 22301.8632812, -7514.8212891, 20520.4863281, -28656.9355469, 29816.6816406
4: -25976.0585938, 15398.0332031, -23905.1875000, 14216.1181641, -40192.1757812, 39303.2187500

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8069665
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8093057
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -20050.4082031, 14754.8818359, -33180.9296875, 33664.2265625
1: -14914.9316406, 13181.6396484, -16229.7929688, 14288.9218750, -29203.8515625, 29411.4316406
2: -21765.7910156, 14396.5722656, -23660.3769531, 15599.0976562, -37364.8867188, 38056.9492188
3: -7514.8212891, 20520.4863281, -8136.4487305, 22301.8632812, -29816.6796875, 28656.9355469
4: -23905.1875000, 14216.1181641, -25976.0585938, 15398.0332031, -39303.2187500, 40192.1757812

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2403169, upper bound: 19342.8038874
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2403169, upper bound: 19342.8063035
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -20050.4082031, 14754.8818359, -34805.2890625, 34805.2851562
1: -16229.7929688, 14288.9218750, -16229.7929688, 14288.9218750, -30518.7148438, 30518.7148438
2: -23660.3769531, 15599.0976562, -23660.3769531, 15599.0976562, -39259.4765625, 39259.4765625
3: -8136.4487305, 22301.8632812, -8136.4487305, 22301.8632812, -30438.3125000, 30438.3125000
4: -25976.0585938, 15398.0332031, -25976.0585938, 15398.0332031, -41374.0937500, 41374.0937500

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7964043
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7976104
time: 0.88 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.73 seconds
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335634
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335634
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335598
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9758356, upper bound: 19343.0311808
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335598
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9758356, upper bound: 19343.0311808
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0379924, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0326363, upper bound: 19342.9798127
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0379924, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0326363, upper bound: 19342.9798127
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0379514, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0311808, upper bound: 19342.9758356
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0379514, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19343.0311808, upper bound: 19342.9758356
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9790818, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9737223, upper bound: 19342.9762439
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9790818, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9737223, upper bound: 19342.9762439
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9790414, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9722668, upper bound: 19342.9722668
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9790414, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9722668, upper bound: 19342.9722668
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19341.9920384, upper bound: 19342.7181527
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19341.9863915, upper bound: 19342.7143948
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19341.9917611, upper bound: 19342.6587280
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19341.9850191, upper bound: 19342.6192967
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8602283, upper bound: 19342.9113639
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8539360, upper bound: 19342.8802317
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8602283, upper bound: 19342.9278552
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8539360, upper bound: 19342.9033768
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8599095, upper bound: 19342.9113639
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8514304, upper bound: 19342.8714795
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8599095, upper bound: 19342.9114108
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8514304, upper bound: 19342.8714794
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19340.6841296, upper bound: 19341.6951246
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19340.6841296, upper bound: 19341.6951246
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8220282, upper bound: 19342.4986184
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.7104673, upper bound: 19342.2931491
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.8220282, upper bound: 19342.4986184
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.7104673, upper bound: 19342.2931491
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19340.8677600, upper bound: 19342.2233612
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19340.8677600, upper bound: 19342.2233612
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19340.8680237, upper bound: 19342.2246841
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.73
Output dim: 3, lower bound: -19340.8680237, upper bound: 19342.2246841
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8819983
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9628283, upper bound: 19342.8598215
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8819983
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9628283, upper bound: 19342.8598215
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8851536
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9626911, upper bound: 19342.8598215
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8851536
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.9626911, upper bound: 19342.8598215
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5067550, upper bound: 19342.7803949
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5067610, upper bound: 19342.7840574
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5067550, upper bound: 19342.8784846
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5067610, upper bound: 19342.8813183
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.6944202
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.7028136
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8069665
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8093057
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.2403169, upper bound: 19342.8038874
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.2403169, upper bound: 19342.8063035
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7964043
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.73
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7976104

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14485.6855469, 10732.5957031, -15482.4042969, 11507.3535156, -25993.0371094, 26215.0000000
1: -11723.9082031, 10387.5712891, -12532.0966797, 11141.1728516, -22865.0800781, 22919.6660156
2: -17096.9316406, 11348.8613281, -18279.1914062, 12169.2402344, -29266.1718750, 29628.0507812
3: -5922.4228516, 16098.1152344, -6347.8457031, 17254.7207031, -23177.1425781, 22445.9609375
4: -18788.8964844, 11219.3486328, -20080.5488281, 12020.0205078, -30808.9160156, 31299.8945312

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16657.4628906, 12336.3632812, -15450.6328125, 11483.5410156, -28141.0039062, 27786.9960938
1: -13484.9306641, 11939.6171875, -12506.4179688, 11118.1708984, -24603.0976562, 24446.0332031
2: -19663.8789062, 13043.1787109, -18241.7480469, 12144.0673828, -31807.9433594, 31284.9257812
3: -6806.5585938, 18534.3945312, -6334.4487305, 17219.2480469, -24025.8066406, 24868.8437500
4: -21611.0488281, 12901.0048828, -20039.3828125, 11995.1171875, -33606.1640625, 32940.3867188

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2654123, upper bound: 19340.5496110
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14485.6855469, 10732.5957031, -16034.9892578, 11901.4921875, -26387.1738281, 26767.5859375
1: -11723.9082031, 10387.5712891, -12979.6025391, 11524.6894531, -23248.5937500, 23367.1718750
2: -17096.9316406, 11348.8613281, -18932.5468750, 12586.2998047, -29683.2304688, 30281.4062500
3: -5922.4228516, 16098.1152344, -6566.6435547, 17867.9726562, -23790.3925781, 22664.7558594
4: -18788.8964844, 11219.3486328, -20799.5703125, 12432.6601562, -31221.5566406, 32018.9160156

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
time: 0.90 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 4.53 seconds
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.53
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.53
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.53
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.53
Output dim: 3, lower bound: -19342.9806214, upper bound: 19343.0345885
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.53
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.53
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9766426, upper bound: 19343.0331330
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335598
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9758356, upper bound: 19343.0311808
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9803802, upper bound: 19343.0335598
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9758356, upper bound: 19343.0311808
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0379924, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0326363, upper bound: 19342.9798127
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0379924, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0326363, upper bound: 19342.9798127
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0379514, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0311808, upper bound: 19342.9758356
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0379514, upper bound: 19342.9835734
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19343.0311808, upper bound: 19342.9758356
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9790818, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9737223, upper bound: 19342.9762439
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9790818, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9737223, upper bound: 19342.9762439
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9790414, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9722668, upper bound: 19342.9722668
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9790414, upper bound: 19342.9800046
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9722668, upper bound: 19342.9722668
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19341.9920384, upper bound: 19342.7181527
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19341.9863915, upper bound: 19342.7143948
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8602283, upper bound: 19342.9113639
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8539360, upper bound: 19342.8802317
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8602283, upper bound: 19342.9278552
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8539360, upper bound: 19342.9033768
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8599095, upper bound: 19342.9113639
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8514304, upper bound: 19342.8714795
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8599095, upper bound: 19342.9114108
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8514304, upper bound: 19342.8714794
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8220282, upper bound: 19342.4986184
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.7104673, upper bound: 19342.2931491
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.8220282, upper bound: 19342.4986184
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.7104673, upper bound: 19342.2931491
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8819983
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9628283, upper bound: 19342.8598215
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8819983
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9628283, upper bound: 19342.8598215
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8851536
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9626911, upper bound: 19342.8598215
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9696417, upper bound: 19342.8851536
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.9626911, upper bound: 19342.8598215
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5067550, upper bound: 19342.7803949
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5067610, upper bound: 19342.7840574
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5067550, upper bound: 19342.8784846
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5067610, upper bound: 19342.8813183
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.6944202
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.7028136
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8069665
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.5016031, upper bound: 19342.8093057
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.2403169, upper bound: 19342.8038874
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.2403169, upper bound: 19342.8063035
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7964043
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.53
Output dim: 3, lower bound: -19342.2378901, upper bound: 19342.7976104
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.5000000, mid=0.5000000, abs_max=24289.171875
rel_dist={3: [-19343.13957058345, 19343.13957058345]}

## Binary search (step 1) starts
Candidate diff: 0.2500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9014902, upper bound: 19343.0711219
time: 0.89 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8949772, upper bound: 19342.8949772
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 3, lower bound: -19342.9014902, upper bound: 19343.0711219
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 3, lower bound: -19342.8949772, upper bound: 19342.8949772

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -15606.2392578, 11546.4335938, -15902.4746094, 11766.7207031, -27372.9609375, 27448.9062500
1: -12631.3779297, 11177.4941406, -12870.9160156, 11389.3173828, -24020.6914062, 24048.4101562
2: -18419.4785156, 12209.6884766, -18770.2070312, 12441.3457031, -30860.8242188, 30979.8945312
3: -6374.2680664, 17344.5468750, -6494.6152344, 17676.2148438, -24050.4824219, 23839.1601562
4: -20242.6933594, 12071.0019531, -20627.9726562, 12301.2226562, -32543.9140625, 32698.9746094

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
time: 0.92 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8800246, upper bound: 19342.9657995
time: 0.81 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -17825.7265625, 13106.9628906, -15762.2304688, 11660.3994141, -29486.1250000, 28869.1933594
1: -14429.7958984, 12691.0000000, -12757.4765625, 11286.0839844, -25715.8789062, 25448.4746094
2: -21053.4453125, 13860.3105469, -18605.5507812, 12328.3027344, -33381.7500000, 32465.8613281
3: -7247.8081055, 19801.2207031, -6435.8588867, 17519.5292969, -24767.3378906, 26237.0800781
4: -23130.5800781, 13698.3564453, -20446.9394531, 12189.3154297, -35319.8906250, 34145.2968750

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.4956349, upper bound: 19342.7211265
time: 0.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8725057
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.38 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -19342.8800246, upper bound: 19342.9657995
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -19342.4956349, upper bound: 19342.7211265
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8725057

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -15549.4179688, 11504.2236328, -15678.2587891, 11600.5712891, -27149.9882812, 27182.4785156
1: -12585.3691406, 11136.6181641, -12689.3115234, 11228.3437500, -23813.7128906, 23825.9238281
2: -18352.5410156, 12165.2158203, -18505.7617188, 12266.0673828, -30618.6093750, 30670.9765625
3: -6351.1728516, 17281.1015625, -6403.5170898, 17425.8847656, -23777.0546875, 23684.6191406
4: -20169.1484375, 12026.9257812, -20337.4550781, 12127.6171875, -32296.7617188, 32364.3769531

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2007932, upper bound: 19342.6167866
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7778539
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
time: 0.88 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -15448.3701172, 11432.0312500, -16400.2597656, 12170.9160156, -27619.2832031, 27832.2910156
1: -12503.2255859, 11067.3828125, -13275.1171875, 11784.9277344, -24288.1503906, 24342.5000000
2: -18235.0058594, 12090.0566406, -19364.8574219, 12871.9023438, -31106.9082031, 31454.9140625
3: -6311.4653320, 17171.6718750, -6716.4560547, 18275.4335938, -24586.8984375, 23888.1289062
4: -20039.4023438, 11951.9736328, -21274.2636719, 12715.4853516, -32754.8867188, 33226.2343750

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8050058, upper bound: 19342.9609074
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8800246, upper bound: 19342.9657995
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8800246, upper bound: 19342.9657995
time: 1.00 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -17769.7324219, 13064.8857422, -15530.5556641, 11488.2998047, -29258.0312500, 28595.4414062
1: -14384.3750000, 12650.4833984, -12569.7861328, 11119.4394531, -25503.8144531, 25220.2675781
2: -20987.2792969, 13816.0302734, -18332.1191406, 12146.7832031, -33134.0625000, 32148.1425781
3: -7225.0317383, 19738.7285156, -6341.7827148, 17260.8125000, -24485.8437500, 26080.5117188
4: -23057.9570312, 13654.3710938, -20146.6171875, 12009.5634766, -35067.5195312, 33800.9843750

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.1981426, upper bound: 19342.5667499
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.1938515, upper bound: 19342.5190790
time: 0.92 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -17676.8515625, 12999.4443359, -16248.2548828, 12057.9960938, -29734.8476562, 29247.6992188
1: -14309.2080078, 12587.6689453, -13152.3398438, 11674.9013672, -25984.1093750, 25740.0058594
2: -20879.5976562, 13747.8388672, -19186.5429688, 12751.4042969, -33631.0000000, 32934.3828125
3: -7188.0625000, 19637.8281250, -6653.0346680, 18106.2304688, -25294.2890625, 26290.8632812
4: -22938.6054688, 13586.5166016, -21077.9960938, 12596.1630859, -35534.7695312, 34664.5039062

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8003660, upper bound: 19342.8672971
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
time: 1.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7778539
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.8800246, upper bound: 19342.9657995
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.8800246, upper bound: 19342.9657995
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.1981426, upper bound: 19342.5667499
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.1938515, upper bound: 19342.5190790
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.8003660, upper bound: 19342.8672971
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.07
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -15678.2587891, 11600.5712891, -26985.6679688, 27061.0625000
1: -12452.2431641, 11018.8076172, -12689.3115234, 11228.3437500, -23680.5820312, 23708.1191406
2: -18158.9550781, 12037.1357422, -18505.7617188, 12266.0673828, -30425.0234375, 30542.8984375
3: -6284.6875000, 17098.0156250, -6403.5170898, 17425.8847656, -23710.5703125, 23501.5332031
4: -19956.5410156, 11899.9658203, -20337.4550781, 12127.6171875, -32084.1542969, 32237.4179688

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7778539
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7778539
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -15678.2587891, 11600.5712891, -27728.2578125, 27648.5039062
1: -13054.4599609, 11591.3388672, -12689.3115234, 11228.3437500, -24282.8046875, 24280.6484375
2: -19041.2832031, 12659.5800781, -18505.7617188, 12266.0673828, -31307.3515625, 31165.3398438
3: -6605.4584961, 17970.6660156, -6403.5170898, 17425.8847656, -24031.3437500, 24374.1835938
4: -20919.1621094, 12504.9296875, -20337.4550781, 12127.6171875, -33046.7773438, 32842.3828125

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -16400.2597656, 12170.9160156, -27556.0097656, 27783.0664062
1: -12452.2431641, 11018.8076172, -13275.1171875, 11784.9277344, -24237.1660156, 24293.9257812
2: -18158.9550781, 12037.1357422, -19364.8574219, 12871.9023438, -31030.8574219, 31401.9921875
3: -6284.6875000, 17098.0156250, -6716.4560547, 18275.4335938, -24560.1210938, 23814.4726562
4: -19956.5410156, 11899.9658203, -21274.2636719, 12715.4853516, -32672.0273438, 33174.2304688

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9657995
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9657995
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -16400.2597656, 12170.9160156, -28298.6035156, 28370.5097656
1: -13054.4599609, 11591.3388672, -13275.1171875, 11784.9277344, -24839.3867188, 24866.4570312
2: -19041.2832031, 12659.5800781, -19364.8574219, 12871.9023438, -31913.1855469, 32024.4375000
3: -6605.4584961, 17970.6660156, -6716.4560547, 18275.4335938, -24880.8925781, 24687.1210938
4: -20919.1621094, 12504.9296875, -21274.2636719, 12715.4853516, -33634.6406250, 33779.1953125

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9371290
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9371290
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17412.1035156, 12799.2851562, -16191.7597656, 12016.7695312, -29428.8730469, 28991.0410156
1: -14094.7568359, 12394.7343750, -13106.5654297, 11635.1025391, -25729.8593750, 25501.2968750
2: -20566.6660156, 13535.2187500, -19119.8007812, 12707.7695312, -33274.4335938, 32655.0195312
3: -7080.2246094, 19342.6386719, -6630.0527344, 18043.5371094, -25123.7617188, 25972.6914062
4: -22595.0800781, 13375.2333984, -21004.7304688, 12552.9140625, -35147.9921875, 34379.9648438

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18973.9863281, 13904.0166016, -15994.1328125, 11864.9326172, -30838.9179688, 29898.1484375
1: -15357.1347656, 13469.4189453, -12947.6865234, 11488.0263672, -26845.1582031, 26417.0996094
2: -22389.1914062, 14706.4199219, -18886.5058594, 12547.4003906, -34936.5898438, 33592.9257812
3: -7682.4609375, 21058.9296875, -6545.2724609, 17820.1582031, -25502.6152344, 27604.1992188
4: -24588.6914062, 14526.2929688, -20746.3554688, 12392.9707031, -36981.6601562, 35272.6484375

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
time: 0.93 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7778539
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7778539
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793756
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9657995
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9657995
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9371290
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.9371290
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.24
Output dim: 3, lower bound: -19342.7957262, upper bound: 19342.7957262

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -15385.0976562, 11382.8056641, -26767.9003906, 26767.9023438
1: -12452.2431641, 11018.8076172, -12452.2431641, 11018.8076172, -23471.0507812, 23471.0507812
2: -18158.9550781, 12037.1357422, -18158.9550781, 12037.1357422, -30196.0898438, 30196.0898438
3: -6284.6875000, 17098.0156250, -6284.6875000, 17098.0156250, -23382.7031250, 23382.7031250
4: -19956.5410156, 11899.9658203, -19956.5410156, 11899.9658203, -31856.5078125, 31856.5058594

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6690235, upper bound: 19341.8300557
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.3907462, upper bound: 19340.3907462
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -17610.9257812, 12945.5888672, -28330.6875000, 28993.7304688
1: -12452.2431641, 11018.8076172, -14255.5488281, 12535.5800781, -24987.8183594, 25274.3554688
2: -18158.9550781, 12037.1357422, -20799.7832031, 13690.3691406, -31849.3242188, 32836.9179688
3: -6284.6875000, 17098.0156250, -7160.5898438, 19561.6269531, -25846.3125000, 24258.6035156
4: -19956.5410156, 11899.9658203, -22852.2402344, 13529.6064453, -33486.1367188, 34752.2031250

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6690235, upper bound: 19342.6447530
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.3907462, upper bound: 19340.6427088
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -15385.0976562, 11382.8056641, -27510.4921875, 27355.3437500
1: -13054.4599609, 11591.3388672, -12452.2431641, 11018.8076172, -24073.2675781, 24043.5800781
2: -19041.2832031, 12659.5800781, -18158.9550781, 12037.1357422, -31078.4179688, 30818.5332031
3: -6605.4584961, 17970.6660156, -6284.6875000, 17098.0156250, -23703.4746094, 24255.3535156
4: -20919.1621094, 12504.9296875, -19956.5410156, 11899.9658203, -32819.1250000, 32461.4707031

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6670337, upper bound: 19341.6818127
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.4734813, upper bound: 19342.7743984
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793755
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -17610.9257812, 12945.5888672, -29073.2773438, 29581.1738281
1: -13054.4599609, 11591.3388672, -14255.5488281, 12535.5800781, -25590.0390625, 25846.8867188
2: -19041.2832031, 12659.5800781, -20799.7832031, 13690.3691406, -32731.6503906, 33459.3632812
3: -6605.4584961, 17970.6660156, -7160.5898438, 19561.6269531, -26167.0859375, 25131.2558594
4: -20919.1621094, 12504.9296875, -22852.2402344, 13529.6064453, -34448.7617188, 35357.1718750

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6670337, upper bound: 19342.6167866
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.4734813, upper bound: 19342.7743984
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793755
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -16127.6875000, 11970.2519531, -27355.3437500, 27510.4921875
1: -12452.2431641, 11018.8076172, -13054.4599609, 11591.3388672, -24043.5800781, 24073.2675781
2: -18158.9550781, 12037.1357422, -19041.2832031, 12659.5800781, -30818.5332031, 31078.4179688
3: -6284.6875000, 17098.0156250, -6605.4584961, 17970.6660156, -24255.3535156, 23703.4746094
4: -19956.5410156, 11899.9658203, -20919.1621094, 12504.9296875, -32461.4707031, 32819.1289062

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8801870, upper bound: 19343.0171124
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8800980, upper bound: 19343.0171124
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15385.0976562, 11382.8056641, -18657.2773438, 13784.1298828, -29169.2246094, 30040.0820312
1: -12452.2431641, 11018.8076172, -15102.3857422, 13346.4589844, -25798.6953125, 26121.1933594
2: -18158.9550781, 12037.1357422, -22039.4648438, 14576.5332031, -32735.4882812, 34076.6015625
3: -6284.6875000, 17098.0156250, -7609.1791992, 20777.9628906, -27062.6484375, 24707.1953125
4: -19956.5410156, 11899.9658203, -24205.4062500, 14394.3369141, -34350.8750000, 36105.3710938

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8801870, upper bound: 19343.0171124
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8800980, upper bound: 19343.0171124
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -16127.6875000, 11970.2519531, -28097.9375000, 28097.9375000
1: -13054.4599609, 11591.3388672, -13054.4599609, 11591.3388672, -24645.7988281, 24645.7988281
2: -19041.2832031, 12659.5800781, -19041.2832031, 12659.5800781, -31700.8613281, 31700.8632812
3: -6605.4584961, 17970.6660156, -6605.4584961, 17970.6660156, -24576.1250000, 24576.1250000
4: -20919.1621094, 12504.9296875, -20919.1621094, 12504.9296875, -33424.0937500, 33424.0898438

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16127.6875000, 11970.2519531, -18657.2773438, 13784.1298828, -29911.8164062, 30627.5273438
1: -13054.4599609, 11591.3388672, -15102.3857422, 13346.4589844, -26400.9160156, 26693.7226562
2: -19041.2832031, 12659.5800781, -22039.4648438, 14576.5332031, -33617.8125000, 34699.0429688
3: -6605.4584961, 17970.6660156, -7609.1791992, 20777.9628906, -27383.4218750, 25579.8457031
4: -20919.1621094, 12504.9296875, -24205.4062500, 14394.3369141, -35313.4921875, 36710.3359375

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17412.1035156, 12799.2851562, -15921.8466797, 11819.5517578, -29231.6562500, 28721.1328125
1: -14094.7568359, 12394.7343750, -12887.7998047, 11444.6542969, -25539.4101562, 25282.5351562
2: -20566.6660156, 13535.2187500, -18800.8652344, 12499.0917969, -33065.7500000, 32336.0820312
3: -7080.2246094, 19342.6386719, -6520.4384766, 17744.0644531, -24824.2890625, 25863.0781250
4: -22595.0800781, 13375.2333984, -20654.7636719, 12346.1992188, -34941.2812500, 34029.9921875

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.4894068
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.8631550
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17412.1035156, 12799.2851562, -17645.0625000, 13033.6357422, -30445.7343750, 30444.3457031
1: -14094.7568359, 12394.7343750, -14283.9912109, 12624.9628906, -26719.7187500, 26678.7265625
2: -20566.6660156, 13535.2187500, -20811.9941406, 13782.2275391, -34348.8906250, 34347.2070312
3: -7080.2246094, 19342.6386719, -7189.7602539, 19628.0117188, -26708.2363281, 26532.3984375
4: -22595.0800781, 13375.2333984, -22852.7089844, 13607.9052734, -36202.9843750, 36227.9414062

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.4894068
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.8631550
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18973.9863281, 13904.0166016, -15921.8466797, 11819.5517578, -30793.5390625, 29825.8632812
1: -15357.1347656, 13469.4189453, -12887.7998047, 11444.6542969, -26801.7890625, 26357.2128906
2: -22389.1914062, 14706.4199219, -18800.8652344, 12499.0917969, -34888.2812500, 33507.2851562
3: -7682.4609375, 21058.9296875, -6520.4384766, 17744.0644531, -25426.5214844, 27579.3671875
4: -24588.6914062, 14526.2929688, -20654.7636719, 12346.1992188, -36934.8906250, 35181.0546875

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.1938514
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.7914749
time: 0.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18973.9863281, 13904.0166016, -17645.0625000, 13033.6357422, -32007.6210938, 31549.0781250
1: -15357.1347656, 13469.4189453, -14283.9912109, 12624.9628906, -27982.0976562, 27753.4062500
2: -22389.1914062, 14706.4199219, -20811.9941406, 13782.2275391, -36171.4179688, 35518.4140625
3: -7682.4609375, 21058.9296875, -7189.7602539, 19628.0117188, -27310.4687500, 28248.6894531
4: -24588.6914062, 14526.2929688, -22852.7089844, 13607.9052734, -38196.5976562, 37379.0000000

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.1938514
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.7914749
time: 0.92 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.62 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19340.6690235, upper bound: 19341.8300557
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19340.3907462, upper bound: 19340.3907462
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19340.6690235, upper bound: 19342.6447530
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19340.3907462, upper bound: 19340.6427088
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.4734813, upper bound: 19342.7743984
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793755
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.4734813, upper bound: 19342.7743984
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.5017222, upper bound: 19342.7793755
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8801870, upper bound: 19343.0171124
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8800980, upper bound: 19343.0171124
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8801870, upper bound: 19343.0171124
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8800980, upper bound: 19343.0171124
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.8742409, upper bound: 19342.9371290
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.4894068
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.8631550
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.4894068
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.8631550
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.1938514
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.7914749
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.1938514
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.62
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.7914749

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -15244.2285156, 11282.8955078, -26765.3007812, 26751.5820312
1: -12532.0966797, 11141.1728516, -12338.0068359, 10921.9160156, -23454.0039062, 23479.1777344
2: -18279.1914062, 12169.2402344, -17992.8847656, 11931.7333984, -30210.9257812, 30162.1250000
3: -6347.8457031, 17254.7207031, -6227.9804688, 16942.3652344, -23290.2089844, 23482.6992188
4: -20080.5488281, 12020.0205078, -19773.4277344, 11795.2558594, -31875.8046875, 31793.4472656

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0213330, upper bound: 19342.9705395
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0189354, upper bound: 19342.9666579
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -15352.4257812, 11358.1484375, -27393.1367188, 27253.9179688
1: -12979.6025391, 11524.6894531, -12425.9160156, 10994.8310547, -23974.4335938, 23950.6054688
2: -18932.5468750, 12586.2998047, -18120.5878906, 12010.8896484, -30943.4375000, 30706.8867188
3: -6566.6435547, 17867.9726562, -6271.2163086, 17061.6210938, -23628.2636719, 24139.1894531
4: -20799.5703125, 12432.6601562, -19914.3398438, 11874.0869141, -32673.6562500, 32347.0000000

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0187603, upper bound: 19342.9660032
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0163868, upper bound: 19342.9622488
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -17475.4941406, 12850.1142578, -28332.5195312, 28982.8457031
1: -12532.0966797, 11141.1728516, -14145.7900391, 12443.0322266, -24975.1269531, 25286.9628906
2: -18279.1914062, 12169.2402344, -20640.1738281, 13589.7031250, -31868.8945312, 32809.4140625
3: -6347.8457031, 17254.7207031, -7106.4804688, 19412.3457031, -25760.1894531, 24361.2011719
4: -20080.5488281, 12020.0205078, -22676.2890625, 13429.5283203, -33510.0781250, 34696.3046875

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.4078229, upper bound: 19342.7212902
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2279141, upper bound: 19342.6376880
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -17572.9023438, 12916.4287109, -28951.4179688, 29474.3925781
1: -12979.6025391, 11524.6894531, -14224.8867188, 12507.0986328, -25486.6953125, 25749.5703125
2: -18932.5468750, 12586.2998047, -20755.0371094, 13659.3476562, -32591.8945312, 33341.3320312
3: -6566.6435547, 17867.9726562, -7144.6489258, 19518.7285156, -26085.3710938, 25012.6210938
4: -20799.5703125, 12432.6601562, -22803.0078125, 13499.0517578, -34298.6210938, 35235.6679688

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.4156075, upper bound: 19342.7197857
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2273225, upper bound: 19342.6359148
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14714.6064453, 10904.2587891, -15986.2011719, 11868.3701172, -26582.9746094, 26890.4609375
1: -11909.0263672, 10553.8779297, -12939.9169922, 11492.3632812, -23401.3906250, 23493.7929688
2: -17367.7363281, 11530.3730469, -18874.4062500, 12551.9941406, -29919.7304688, 30404.7792969
3: -6016.9643555, 16355.0917969, -6548.4926758, 17813.7050781, -23830.6699219, 22903.5820312
4: -19086.1601562, 11398.5839844, -20735.1191406, 12398.3017578, -31484.4609375, 32133.7011719

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.5380203, upper bound: 19340.3343996
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9721921, upper bound: 19343.0249615
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9631046, upper bound: 19343.0184324
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -15291.7265625, 11311.9248047, -16094.2050781, 11945.5488281, -27237.2753906, 27406.1289062
1: -12376.9931641, 10950.3818359, -13027.4228516, 11567.3701172, -23944.3632812, 23977.8046875
2: -18049.3437500, 11962.5625000, -19002.0000000, 12633.2763672, -30682.6210938, 30964.5585938
3: -6245.9423828, 16994.0996094, -6591.5366211, 17933.6250000, -24179.5664062, 23585.6347656
4: -19835.9296875, 11825.4414062, -20875.9453125, 12478.9824219, -32314.9121094, 32701.3867188

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.6804196, upper bound: 19340.6669326
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9709634, upper bound: 19343.0228891
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9622482, upper bound: 19343.0163868
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14714.6064453, 10904.2587891, -18523.0332031, 13689.0576172, -28403.6562500, 29427.2929688
1: -11909.0263672, 10553.8779297, -14993.7265625, 13253.6591797, -25162.6835938, 25547.6015625
2: -17367.7363281, 11530.3730469, -21881.2324219, 14476.3837891, -31844.1191406, 33411.6015625
3: -6016.9643555, 16355.0917969, -7555.4077148, 20629.7968750, -26646.7617188, 23910.5000000
4: -19086.1601562, 11398.5839844, -24030.7851562, 14294.5693359, -33380.7265625, 35429.3671875

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.5172098, upper bound: 19340.3334682
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.5080481, upper bound: 19340.3334815
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15291.7265625, 11311.9248047, -18618.9296875, 13754.6904297, -29046.4160156, 29930.8554688
1: -12376.9931641, 10950.3818359, -15071.3828125, 13318.3740234, -25695.3632812, 26021.7656250
2: -18049.3437500, 11962.5625000, -21994.3808594, 14545.6523438, -32594.9960938, 33956.9453125
3: -6245.9423828, 16994.0996094, -7592.9960938, 20734.9941406, -26980.9375000, 24587.0957031
4: -19835.9296875, 11825.4414062, -24155.8847656, 14363.4853516, -34199.4140625, 35981.3281250

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.6549994, upper bound: 19340.6648671
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.6432030, upper bound: 19340.6640714
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -15986.2011719, 11868.3701172, -27350.7734375, 27493.5546875
1: -12532.0966797, 11141.1728516, -12939.9169922, 11492.3632812, -24024.4589844, 24081.0898438
2: -18279.1914062, 12169.2402344, -18874.4062500, 12551.9941406, -30831.1855469, 31043.6464844
3: -6347.8457031, 17254.7207031, -6548.4926758, 17813.7050781, -24161.5507812, 23803.2109375
4: -20080.5488281, 12020.0205078, -20735.1191406, 12398.3017578, -32478.8515625, 32755.1386719

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9708751, upper bound: 19342.9694118
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9623964, upper bound: 19342.9649120
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -16094.2050781, 11945.5488281, -27980.5390625, 27995.6953125
1: -12979.6025391, 11524.6894531, -13027.4228516, 11567.3701172, -24546.9726562, 24552.1093750
2: -18932.5468750, 12586.2998047, -19002.0000000, 12633.2763672, -31565.8242188, 31588.2968750
3: -6566.6435547, 17867.9726562, -6591.5366211, 17933.6250000, -24500.2656250, 24459.5058594
4: -20799.5703125, 12432.6601562, -20875.9453125, 12478.9824219, -33278.5546875, 33308.6054688

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9689314, upper bound: 19342.9652147
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9608228, upper bound: 19342.9608228
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -18523.0332031, 13689.0576172, -29171.4609375, 30030.3867188
1: -12532.0966797, 11141.1728516, -14993.7265625, 13253.6591797, -25785.7519531, 26134.8984375
2: -18279.1914062, 12169.2402344, -21881.2324219, 14476.3837891, -32755.5742188, 34050.4726562
3: -6347.8457031, 17254.7207031, -7555.4077148, 20629.7968750, -26977.6406250, 24810.1289062
4: -20080.5488281, 12020.0205078, -24030.7851562, 14294.5693359, -34375.1171875, 36050.8046875

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8699753, upper bound: 19342.9120742
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8455143, upper bound: 19342.8940224
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -18618.9296875, 13754.6904297, -29789.6796875, 30520.4199219
1: -12979.6025391, 11524.6894531, -15071.3828125, 13318.3740234, -26297.9726562, 26596.0722656
2: -18932.5468750, 12586.2998047, -21994.3808594, 14545.6523438, -33478.1992188, 34580.6796875
3: -6566.6435547, 17867.9726562, -7592.9960938, 20734.9941406, -27301.6347656, 25460.9687500
4: -20799.5703125, 12432.6601562, -24155.8847656, 14363.4853516, -35163.0546875, 36588.5468750

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8666954, upper bound: 19342.9076063
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -15921.8466797, 11819.5517578, -30245.6015625, 29535.6640625
1: -14914.9316406, 13181.6396484, -12887.7998047, 11444.6542969, -26359.5839844, 26069.4335938
2: -21765.7910156, 14396.5722656, -18800.8652344, 12499.0917969, -34264.8828125, 33197.4335938
3: -7514.8212891, 20520.4863281, -6520.4384766, 17744.0644531, -25258.8828125, 27040.9257812
4: -23905.1875000, 14216.1181641, -20654.7636719, 12346.1992188, -36251.3867188, 34870.8750000

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.4309752, upper bound: 19342.8721501
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.4309752, upper bound: 19342.8721501
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -17645.0625000, 13033.6357422, -31459.6855469, 31258.8808594
1: -14914.9316406, 13181.6396484, -14283.9912109, 12624.9628906, -27539.8945312, 27465.6269531
2: -21765.7910156, 14396.5722656, -20811.9941406, 13782.2275391, -35548.0117188, 35208.5625000
3: -7514.8212891, 20520.4863281, -7189.7602539, 19628.0117188, -27142.8300781, 27710.2460938
4: -23905.1875000, 14216.1181641, -22852.7089844, 13607.9052734, -37513.0937500, 37068.8281250

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.5678101, upper bound: 19342.8449699
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.8631550
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -15921.8466797, 11819.5517578, -31869.9609375, 30676.7265625
1: -16229.7929688, 14288.9218750, -12887.7998047, 11444.6542969, -27674.4472656, 27176.7226562
2: -23660.3769531, 15599.0976562, -18800.8652344, 12499.0917969, -36159.4687500, 34399.9609375
3: -8136.4487305, 22301.8632812, -6520.4384766, 17744.0644531, -25880.5136719, 28822.3007812
4: -25976.0585938, 15398.0332031, -20654.7636719, 12346.1992188, -38322.2578125, 36052.7929688

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2979992, upper bound: 19342.7999659
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.2979992, upper bound: 19342.7999659
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -17645.0625000, 13033.6357422, -33084.0351562, 32399.9433594
1: -16229.7929688, 14288.9218750, -14283.9912109, 12624.9628906, -28854.7558594, 28572.9140625
2: -23660.3769531, 15599.0976562, -20811.9941406, 13782.2275391, -37442.6054688, 36411.0859375
3: -8136.4487305, 22301.8632812, -7189.7602539, 19628.0117188, -27764.4609375, 29491.6230469
4: -25976.0585938, 15398.0332031, -22852.7089844, 13607.9052734, -39583.9648438, 38250.7421875

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.5040661, upper bound: 19342.7852417
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.7914749
time: 0.89 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.13 seconds
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19343.0213330, upper bound: 19342.9705395
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19343.0189354, upper bound: 19342.9666579
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19343.0187603, upper bound: 19342.9660032
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19343.0163868, upper bound: 19342.9622488
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.4078229, upper bound: 19342.7212902
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.2279141, upper bound: 19342.6376880
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.4156075, upper bound: 19342.7197857
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.2273225, upper bound: 19342.6359148
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9721921, upper bound: 19343.0249615
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9631046, upper bound: 19343.0184324
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9709634, upper bound: 19343.0228891
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9622482, upper bound: 19343.0163868
IS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 3, lower bound: -19341.5172098, upper bound: 19340.3334682
IS_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 3, lower bound: -19341.5080481, upper bound: 19340.3334815
IS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 3, lower bound: -19341.6549994, upper bound: 19340.6648671
IS_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.13
Output dim: 3, lower bound: -19341.6432030, upper bound: 19340.6640714
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9708751, upper bound: 19342.9694118
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9623964, upper bound: 19342.9649120
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9689314, upper bound: 19342.9652147
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.9608228, upper bound: 19342.9608228
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.8699753, upper bound: 19342.9120742
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.8455143, upper bound: 19342.8940224
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.8666954, upper bound: 19342.9076063
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.4309752, upper bound: 19342.8721501
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.4309752, upper bound: 19342.8721501
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19341.5678101, upper bound: 19342.8449699
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.1656689, upper bound: 19342.8631550
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.2979992, upper bound: 19342.7999659
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.2979992, upper bound: 19342.7999659
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19341.5040661, upper bound: 19342.7852417
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.13
Output dim: 3, lower bound: -19342.0330334, upper bound: 19342.7914749

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -15016.1220703, 11111.7197266, -26594.1250000, 26523.4707031
1: -12532.0966797, 11141.1728516, -12153.6044922, 10756.0595703, -23288.1562500, 23294.7773438
2: -18279.1914062, 12169.2402344, -17723.1640625, 11750.7958984, -30029.9882812, 29892.4042969
3: -6347.8457031, 17254.7207031, -6133.6875000, 16686.3769531, -23034.2226562, 23388.4042969
4: -20080.5488281, 12020.0205078, -19477.3339844, 11616.6103516, -31697.1601562, 31497.3554688

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19339.6926519, upper bound: 19340.6170942
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0189354, upper bound: 19342.9666579
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0189354, upper bound: 19342.9666579
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15393.2587891, 11440.4765625, -17120.1445312, 12671.6953125, -28064.9531250, 28560.6191406
1: -12460.0537109, 11076.5712891, -13859.9238281, 12264.8251953, -24724.8789062, 24936.4960938
2: -18174.1406250, 12098.5312500, -20210.2246094, 13397.7636719, -31571.9042969, 32308.7539062
3: -6310.2026367, 17155.1875000, -6994.6186523, 19050.2890625, -25360.4902344, 24149.8046875
4: -19965.0585938, 11950.0830078, -22212.0371094, 13252.3066406, -33217.3593750, 34162.1210938

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6637552, upper bound: 19341.6743264
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0154595, upper bound: 19342.9619093
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0048347, upper bound: 19342.8024241
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -15124.1748047, 11186.7285156, -27221.7148438, 27025.6640625
1: -12979.6025391, 11524.6894531, -12241.4238281, 10828.7939453, -23808.3925781, 23766.1113281
2: -18932.5468750, 12586.2998047, -17850.7089844, 11829.7050781, -30762.2519531, 30437.0078125
3: -6566.6435547, 17867.9726562, -6176.7802734, 16805.4238281, -23372.0625000, 24044.7480469
4: -20799.5703125, 12432.6601562, -19618.0605469, 11695.1767578, -32494.7441406, 32050.7207031

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0163868, upper bound: 19342.9622488
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0163868, upper bound: 19342.9622488
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15956.3066406, 11842.9355469, -17216.6601562, 12739.7744141, -28696.0820312, 29059.5917969
1: -12915.9775391, 11468.0820312, -13937.9863281, 12330.4033203, -25246.3789062, 25406.0683594
2: -18839.7929688, 12524.3593750, -20324.6347656, 13469.2470703, -32309.0390625, 32848.9921875
3: -6533.9340820, 17780.2539062, -7034.2851562, 19156.4277344, -25690.3613281, 24814.5390625
4: -20697.6425781, 12371.3837891, -22338.3867188, 13323.8642578, -34021.5078125, 34709.7656250

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19340.6630870, upper bound: 19341.6743264
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0136959, upper bound: 19342.9574423
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9500472, upper bound: 19342.7518455
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -17251.3847656, 12682.0361328, -28164.4394531, 28758.7382812
1: -12532.0966797, 11141.1728516, -13964.6240234, 12280.4501953, -24812.5410156, 25105.7949219
2: -18279.1914062, 12169.2402344, -20375.2578125, 13411.8769531, -31691.0683594, 32544.4980469
3: -6347.8457031, 17254.7207031, -7013.8833008, 19161.3906250, -25509.2363281, 24268.6035156
4: -20080.5488281, 12020.0205078, -22385.3320312, 13253.9716797, -33334.5195312, 34405.3515625

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.1093977, upper bound: 19342.5919242
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2279141, upper bound: 19342.6376880
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2279141, upper bound: 19342.6376880
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -17351.4863281, 12750.4941406, -28785.4843750, 29252.9785156
1: -12979.6025391, 11524.6894531, -14045.8583984, 12346.6464844, -25326.2460938, 25570.5468750
2: -18932.5468750, 12586.2998047, -20493.1953125, 13484.0722656, -32416.6191406, 33079.4960938
3: -6566.6435547, 17867.9726562, -7053.2998047, 19270.7714844, -25837.4140625, 24921.2714844
4: -20799.5703125, 12432.6601562, -22515.5312500, 13325.6601562, -34125.2265625, 34948.1914062

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.1146955, upper bound: 19342.5917301
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2273225, upper bound: 19342.6359148
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.2273225, upper bound: 19342.6359148
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14714.6064453, 10904.2587891, -15759.6269531, 11699.0488281, -26413.6542969, 26663.8847656
1: -11909.0263672, 10553.8779297, -12756.6972656, 11328.7353516, -23237.7578125, 23310.5742188
2: -17367.7363281, 11530.3730469, -18606.4257812, 12373.1035156, -29740.8398438, 30136.7988281
3: -6016.9643555, 16355.0917969, -6454.7070312, 17559.9023438, -23576.8671875, 22809.7968750
4: -19086.1601562, 11398.5839844, -20440.9648438, 12221.8476562, -31308.0058594, 31839.5468750

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9631046, upper bound: 19343.0184324
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9631046, upper bound: 19343.0184324
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14623.8730469, 10835.9560547, -17826.6933594, 13224.8486328, -27848.7187500, 28662.6484375
1: -11835.6445312, 10487.8974609, -14433.2353516, 12807.5273438, -24643.1718750, 24921.1250000
2: -17260.7597656, 11458.1669922, -21050.2636719, 13988.1064453, -31248.8632812, 32508.4277344
3: -5978.7729492, 16253.6191406, -7300.1835938, 19878.4121094, -25857.1855469, 23553.8007812
4: -18968.5722656, 11327.1630859, -23127.9726562, 13826.8085938, -32795.3828125, 34455.1367188

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9387851, upper bound: 19342.9509293
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9474175, upper bound: 19343.0011334
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -15291.7265625, 11311.9248047, -15867.4531250, 11776.0732422, -27067.8007812, 27179.3789062
1: -12376.9931641, 10950.3818359, -12844.0781250, 11403.5126953, -23780.5000000, 23794.4589844
2: -18049.3437500, 11962.5625000, -18733.8574219, 12454.1884766, -30503.5312500, 30696.4179688
3: -6245.9423828, 16994.0996094, -6497.6352539, 17679.6191406, -23925.5625000, 23491.7343750
4: -19835.9296875, 11825.4414062, -20581.5800781, 12302.3242188, -32138.2539062, 32407.0214844

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9622482, upper bound: 19343.0163868
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9622488, upper bound: 19343.0163868
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15214.0654297, 11253.7919922, -17929.4980469, 13300.5273438, -28514.5937500, 29183.2890625
1: -12314.2001953, 10894.2314453, -14516.1074219, 12881.0341797, -25195.2304688, 25410.3398438
2: -17957.8535156, 11901.3701172, -21172.2187500, 14068.1601562, -32026.0117188, 33073.5898438
3: -6213.7407227, 16907.4433594, -7342.7792969, 19993.1582031, -26206.8984375, 24250.2226562
4: -19735.3886719, 11764.6240234, -23262.7539062, 13906.0566406, -33641.4453125, 35027.3750000

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9383811, upper bound: 19342.9496279
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9464682, upper bound: 19342.9977110
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -15759.6269531, 11699.0488281, -27181.4531250, 27266.9785156
1: -12532.0966797, 11141.1728516, -12756.6972656, 11328.7353516, -23860.8261719, 23897.8710938
2: -18279.1914062, 12169.2402344, -18606.4257812, 12373.1035156, -30652.2949219, 30775.6660156
3: -6347.8457031, 17254.7207031, -6454.7070312, 17559.9023438, -23907.7480469, 23709.4277344
4: -20080.5488281, 12020.0205078, -20440.9648438, 12221.8476562, -32302.3964844, 32460.9843750

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9623964, upper bound: 19342.9649120
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9623964, upper bound: 19342.9649120
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -15393.2587891, 11440.4765625, -17826.6933594, 13224.8486328, -28618.1074219, 29267.1699219
1: -12460.0537109, 11076.5712891, -14433.2353516, 12807.5273438, -25267.5800781, 25509.8046875
2: -18174.1406250, 12098.5312500, -21050.2636719, 13988.1064453, -32162.2441406, 33148.7929688
3: -6310.2026367, 17155.1875000, -7300.1835938, 19878.4121094, -26188.6152344, 24455.3691406
4: -19965.0585938, 11950.0830078, -23127.9726562, 13826.8085938, -33791.8632812, 35078.0507812

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535601, upper bound: 19342.9419163
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7482132, upper bound: 19342.7893500
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -15867.4531250, 11776.0732422, -27811.0625000, 27768.9453125
1: -12979.6025391, 11524.6894531, -12844.0781250, 11403.5126953, -24383.1093750, 24368.7636719
2: -18932.5468750, 12586.2998047, -18733.8574219, 12454.1884766, -31386.7343750, 31320.1562500
3: -6566.6435547, 17867.9726562, -6497.6352539, 17679.6191406, -24246.2597656, 24365.6074219
4: -20799.5703125, 12432.6601562, -20581.5800781, 12302.3242188, -33101.8945312, 33014.2421875

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9608228, upper bound: 19342.9608228
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9608228, upper bound: 19342.9608228
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -15956.3066406, 11842.9355469, -17929.4980469, 13300.5273438, -29256.8339844, 29772.4316406
1: -12915.9775391, 11468.0820312, -14516.1074219, 12881.0341797, -25797.0097656, 25984.1894531
2: -18839.7929688, 12524.3593750, -21172.2187500, 14068.1601562, -32907.9531250, 33696.5781250
3: -6533.9340820, 17780.2539062, -7342.7792969, 19993.1582031, -26527.0917969, 25123.0312500
4: -20697.6425781, 12371.3837891, -23262.7539062, 13906.0566406, -34603.6992188, 35634.1328125

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7521887, upper bound: 19342.9363087
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7417055, upper bound: 19342.7417055
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -15482.4042969, 11507.3535156, -18302.8144531, 13523.4892578, -29005.8925781, 29810.1679688
1: -12532.0966797, 11141.1728516, -14815.7207031, 13093.6181641, -25625.7148438, 25956.8945312
2: -18279.1914062, 12169.2402344, -21620.8496094, 14301.4716797, -32580.6621094, 33790.0898438
3: -6347.8457031, 17254.7207031, -7464.0634766, 20382.9492188, -26730.7949219, 24718.7851562
4: -20080.5488281, 12020.0205078, -23744.8593750, 14122.0585938, -34202.6093750, 35764.8789062

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7979210, upper bound: 19342.8635829
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8455143, upper bound: 19342.8940224
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8455143, upper bound: 19342.8940224
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -15393.2587891, 11440.4765625, -20303.7031250, 14990.1279297, -30383.3867188, 31744.1796875
1: -12460.0537109, 11076.5712891, -16438.7402344, 14516.9736328, -26977.0253906, 27515.3125000
2: -18174.1406250, 12098.5312500, -23985.5957031, 15857.1152344, -34031.2539062, 36084.1210938
3: -6310.2026367, 17155.1875000, -8279.7968750, 22621.1582031, -28931.3613281, 25434.9843750
4: -19965.0585938, 11950.0830078, -26344.9746094, 15668.5576172, -35633.6132812, 38295.0546875

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7882738, upper bound: 19342.8440121
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8182701, upper bound: 19342.8107019
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8282612, upper bound: 19342.8661675
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16034.9892578, 11901.4921875, -18402.5566406, 13592.0625000, -29627.0507812, 30304.0488281
1: -12979.6025391, 11524.6894531, -14896.4335938, 13161.5302734, -26141.1328125, 26421.1210938
2: -18932.5468750, 12586.2998047, -21738.4902344, 14374.4218750, -33306.9687500, 34324.7890625
3: -6566.6435547, 17867.9726562, -7503.3217773, 20492.4960938, -27059.1367188, 25371.2949219
4: -20799.5703125, 12432.6601562, -23874.9140625, 14193.9843750, -34993.5546875, 36307.5742188

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7953535, upper bound: 19342.8554330
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15956.3066406, 11842.9355469, -20412.3671875, 15069.6699219, -31025.9765625, 32255.2988281
1: -12915.9775391, 11468.0820312, -16526.2910156, 14594.1464844, -27510.1230469, 27994.3730469
2: -18839.7929688, 12524.3593750, -24114.0957031, 15941.2734375, -34781.0625000, 36638.4531250
3: -6533.9340820, 17780.2539062, -8324.6210938, 22741.8652344, -29275.7988281, 26104.8750000
4: -20697.6425781, 12371.3837891, -26487.1406250, 15751.9287109, -36449.5703125, 38858.5234375

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7859383, upper bound: 19342.8353665
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8119476, upper bound: 19342.8045930
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8253847, upper bound: 19342.8571131
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -15798.8115234, 11730.4638672, -30156.5136719, 29412.6289062
1: -14914.9316406, 13181.6396484, -12787.9951172, 11359.9042969, -26274.8320312, 25969.6347656
2: -21765.7910156, 14396.5722656, -18652.8398438, 12406.0644531, -34171.8554688, 33049.4062500
3: -7514.8212891, 20520.4863281, -6472.2036133, 17605.9746094, -25120.7929688, 26992.6894531
4: -23905.1875000, 14216.1181641, -20492.8164062, 12253.8300781, -36159.0156250, 34708.9335938

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8724994, upper bound: 19342.8695315
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8721501
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18426.0507812, 13613.8183594, -18426.0507812, 13613.8183594, -32039.8691406, 32039.8691406
1: -14914.9316406, 13181.6396484, -14914.9316406, 13181.6396484, -28096.5625000, 28096.5625000
2: -21765.7910156, 14396.5722656, -21765.7910156, 14396.5722656, -36162.3593750, 36162.3593750
3: -7514.8212891, 20520.4863281, -7514.8212891, 20520.4863281, -28035.3085938, 28035.3066406
4: -23905.1875000, 14216.1181641, -23905.1875000, 14216.1181641, -38121.3046875, 38121.3046875

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8724994, upper bound: 19342.8695315
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8721501
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18293.1113281, 13519.2597656, -17002.8085938, 12570.0429688, -30863.1542969, 30522.0664062
1: -14807.1367188, 13089.7832031, -13765.0019531, 12173.3583984, -26980.4941406, 26854.7851562
2: -21608.7480469, 14296.9570312, -20054.8144531, 13290.2324219, -34898.9804688, 34351.7734375
3: -7461.3168945, 20373.3242188, -6931.2944336, 18915.2558594, -26376.5722656, 27304.6171875
4: -23731.8847656, 14116.8603516, -22018.8261719, 13121.0888672, -36852.9726562, 36135.6757812

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7901639, upper bound: 19342.8449697
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7851911, upper bound: 19342.8102888
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18387.1914062, 13583.9941406, -17542.6718750, 12956.5996094, -31343.7910156, 31126.6660156
1: -14883.5498047, 13153.1035156, -14201.3925781, 12550.3984375, -27433.9433594, 27354.4921875
2: -21720.1113281, 14364.6982422, -20691.7988281, 13700.3437500, -35420.4492188, 35056.4882812
3: -7498.4946289, 20476.8417969, -7146.3896484, 19513.7089844, -27012.2031250, 27623.2304688
4: -23855.0332031, 14184.9619141, -22720.4218750, 13527.1894531, -37382.2226562, 36905.3828125

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7901639, upper bound: 19342.8540992
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7851911, upper bound: 19342.8271444
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -15798.8115234, 11730.4638672, -31780.8691406, 30553.6914062
1: -16229.7929688, 14288.9218750, -12787.9951172, 11359.9042969, -27589.6972656, 27076.9179688
2: -23660.3769531, 15599.0976562, -18652.8398438, 12406.0644531, -36066.4414062, 34251.9257812
3: -8136.4487305, 22301.8632812, -6472.2036133, 17605.9746094, -25742.4238281, 28774.0664062
4: -25976.0585938, 15398.0332031, -20492.8164062, 12253.8300781, -38229.8906250, 35890.8515625

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7974873
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7999659
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -20050.4082031, 14754.8818359, -18426.0507812, 13613.8183594, -33664.2265625, 33180.9296875
1: -16229.7929688, 14288.9218750, -14914.9316406, 13181.6396484, -29411.4335938, 29203.8515625
2: -23660.3769531, 15599.0976562, -21765.7910156, 14396.5722656, -38056.9492188, 37364.8867188
3: -8136.4487305, 22301.8632812, -7514.8212891, 20520.4863281, -28656.9355469, 29816.6816406
4: -25976.0585938, 15398.0332031, -23905.1875000, 14216.1181641, -40192.1757812, 39303.2187500

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7974873
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7999659
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -19933.5039062, 14669.5722656, -17002.8085938, 12570.0429688, -32503.5468750, 31672.3808594
1: -16135.4189453, 14206.2939453, -13765.0019531, 12173.3583984, -28308.7734375, 27971.2968750
2: -23522.8222656, 15509.1962891, -20054.8144531, 13290.2324219, -36813.0546875, 35564.0117188
3: -8088.1694336, 22171.3789062, -6931.2944336, 18915.2558594, -27003.4238281, 29102.6738281
4: -25823.9062500, 15308.5302734, -22018.8261719, 13121.0888672, -38944.9960938, 37327.3554688

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7856997, upper bound: 19342.7849397
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7819080, upper bound: 19342.7732374
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -20003.4667969, 14720.5927734, -17542.6718750, 12956.5996094, -32960.0664062, 32263.2636719
1: -16191.8466797, 14255.6113281, -14201.3925781, 12550.3984375, -28742.2441406, 28457.0019531
2: -23605.2207031, 15562.5869141, -20691.7988281, 13700.3437500, -37305.5585938, 36254.3789062
3: -8117.6079102, 22250.5156250, -7146.3896484, 19513.7089844, -27631.3164062, 29396.9062500
4: -25915.5585938, 15362.3085938, -22720.4218750, 13527.1894531, -39442.7500000, 38082.7304688

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7856997, upper bound: 19342.7864396
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7819080, upper bound: 19342.7756401
time: 1.09 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.94 seconds
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19343.0189354, upper bound: 19342.9666579
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19343.0189354, upper bound: 19342.9666579
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19343.0154595, upper bound: 19342.9619093
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19343.0048347, upper bound: 19342.8024241
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19343.0163868, upper bound: 19342.9622488
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19343.0163868, upper bound: 19342.9622488
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19343.0136959, upper bound: 19342.9574423
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9500472, upper bound: 19342.7518455
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.2279141, upper bound: 19342.6376880
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.2279141, upper bound: 19342.6376880
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.2273225, upper bound: 19342.6359148
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.2273225, upper bound: 19342.6359148
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9631046, upper bound: 19343.0184324
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9631046, upper bound: 19343.0184324
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9387851, upper bound: 19342.9509293
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9474175, upper bound: 19343.0011334
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9622482, upper bound: 19343.0163868
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9622488, upper bound: 19343.0163868
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9383811, upper bound: 19342.9496279
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9464682, upper bound: 19342.9977110
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9623964, upper bound: 19342.9649120
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9623964, upper bound: 19342.9649120
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7535601, upper bound: 19342.9419163
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7482132, upper bound: 19342.7893500
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9608228, upper bound: 19342.9608228
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.9608228, upper bound: 19342.9608228
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7521887, upper bound: 19342.9363087
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7417055, upper bound: 19342.7417055
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8455143, upper bound: 19342.8940224
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8455143, upper bound: 19342.8940224
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8182701, upper bound: 19342.8107019
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8282612, upper bound: 19342.8661675
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8119476, upper bound: 19342.8045930
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8253847, upper bound: 19342.8571131
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8724994, upper bound: 19342.8695315
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8721501
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8724994, upper bound: 19342.8695315
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8721501
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7901639, upper bound: 19342.8449697
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7851911, upper bound: 19342.8102888
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7901639, upper bound: 19342.8540992
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7851911, upper bound: 19342.8271444
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7974873
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7999659
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7974873
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7999659
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7856997, upper bound: 19342.7849397
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7819080, upper bound: 19342.7732374
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7856997, upper bound: 19342.7864396
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.94
Output dim: 3, lower bound: -19342.7819080, upper bound: 19342.7756401

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15255.0937500, 11337.3496094, -15016.1220703, 11111.7197266, -26366.8125000, 26353.4667969
1: -12348.2080078, 10977.1308594, -12153.6044922, 10756.0595703, -23104.2656250, 23130.7343750
2: -18010.2109375, 11989.8974609, -17723.1640625, 11750.7958984, -29761.0078125, 29713.0605469
3: -6253.8476562, 17000.0175781, -6133.6875000, 16686.3769531, -22940.2226562, 23133.6992188
4: -19785.3125000, 11843.1386719, -19477.3339844, 11616.6103516, -31401.9218750, 31320.4726562

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9222212, upper bound: 19342.9652583
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0205696, upper bound: 19342.9681771
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17397.1445312, 12911.1464844, -15016.1220703, 11111.7197266, -28508.8632812, 27927.2636719
1: -14084.9609375, 12503.2597656, -12153.6044922, 10756.0595703, -24841.0195312, 24656.8632812
2: -20542.0156250, 13656.3964844, -17723.1640625, 11750.7958984, -32292.8125000, 31379.5605469
3: -7125.1450195, 19397.3964844, -6133.6875000, 16686.3769531, -23811.5214844, 25531.0800781
4: -22568.6523438, 13498.4541016, -19477.3339844, 11616.6103516, -34185.2578125, 32975.7851562

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9222212, upper bound: 19342.9652583
time: 7.01 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0205696, upper bound: 19342.9681771
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14929.6875000, 11099.9179688, -16980.7050781, 12572.5576172, -27502.2402344, 28080.6230469
1: -12085.0156250, 10747.1035156, -13746.8515625, 12168.5878906, -24253.6035156, 24493.9550781
2: -17623.0000000, 11740.6933594, -20043.8046875, 13292.9531250, -30915.9531250, 31784.4921875
3: -6122.3735352, 16635.3027344, -6939.6831055, 18894.7363281, -25017.1093750, 23574.9863281
4: -19360.4355469, 11595.5937500, -22029.6562500, 13149.0292969, -32509.4648438, 33625.2500000

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0099132, upper bound: 19342.9262134
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.6696446, upper bound: 19342.8085455
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15304.0664062, 11376.9824219, -17036.5878906, 12607.0791016, -27911.1445312, 28413.5703125
1: -12387.4824219, 11014.5996094, -13792.5625000, 12202.4423828, -24589.9257812, 24807.1621094
2: -18069.0976562, 12031.1328125, -20111.6992188, 13329.6083984, -31398.7070312, 32142.8300781
3: -6276.4975586, 17056.5156250, -6958.2763672, 18955.7539062, -25232.2519531, 24014.7929688
4: -19849.8808594, 11882.6572266, -22103.8281250, 13184.7089844, -33034.5898438, 33986.4804688

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8735318, upper bound: 19342.6832340
time: 2.77 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6429665, upper bound: 19342.5841148
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15808.7578125, 11732.4414062, -15124.1748047, 11186.7285156, -26995.4824219, 26856.6152344
1: -12796.6992188, 11361.1044922, -12241.4238281, 10828.7939453, -23625.4921875, 23602.5273438
2: -18665.0468750, 12407.6552734, -17850.7089844, 11829.7050781, -30494.7519531, 30258.3632812
3: -6472.9467773, 17614.5253906, -6176.7802734, 16805.4238281, -23278.3671875, 23791.3007812
4: -20505.8945312, 12256.4169922, -19618.0605469, 11695.1767578, -32201.0683594, 31874.4765625

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9199390, upper bound: 19342.9606962
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0173755, upper bound: 19342.9637199
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17867.7812500, 13254.6699219, -15124.1748047, 11186.7285156, -29054.5058594, 28378.8437500
1: -14466.2548828, 12836.6572266, -12241.4238281, 10828.7939453, -25295.0468750, 25078.0820312
2: -21099.7460938, 14019.5605469, -17850.7089844, 11829.7050781, -32929.4531250, 31870.2675781
3: -7317.1787109, 19924.7968750, -6176.7802734, 16805.4238281, -24122.6015625, 26101.5742188
4: -23183.0937500, 13857.9882812, -19618.0605469, 11695.1767578, -34878.2656250, 33476.0468750

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9199390, upper bound: 19342.9606962
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19343.0173755, upper bound: 19342.9637199
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15504.6279297, 11512.0810547, -17076.8457031, 12640.4248047, -28145.0527344, 28588.9257812
1: -12550.7431641, 11147.4677734, -13824.6132812, 12233.9433594, -24784.6875000, 24972.0800781
2: -18302.7792969, 12176.4746094, -20157.7949219, 13364.1875000, -31666.9667969, 32334.2695312
3: -6351.2114258, 17273.7890625, -6979.2265625, 19000.4257812, -25351.6367188, 24253.0156250
4: -20108.4980469, 12026.6308594, -22155.5410156, 13220.2421875, -33328.7382812, 34182.1718750

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9464340, upper bound: 19342.9333845
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9952459, upper bound: 19342.9414434
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15841.5400391, 11758.2138672, -17131.6210938, 12674.0439453, -28515.5820312, 28889.8300781
1: -12822.6953125, 11386.1210938, -13869.4316406, 12266.9550781, -25089.6503906, 25255.5527344
2: -18704.2441406, 12434.8437500, -20224.2792969, 13399.9521484, -32104.1914062, 32659.1230469
3: -6488.5131836, 17652.8593750, -6997.3208008, 19060.2480469, -25548.7617188, 24650.1796875
4: -20548.7187500, 12282.1689453, -22228.1660156, 13255.1376953, -33803.8554688, 34510.3320312

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8193656, upper bound: 19342.6816901
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.6017187, upper bound: 19342.5830691
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14485.6855469, 10732.5957031, -15759.6269531, 11699.0488281, -26184.7343750, 26492.2226562
1: -11723.9082031, 10387.5712891, -12756.6972656, 11328.7353516, -23052.6386719, 23144.2695312
2: -17096.9316406, 11348.8613281, -18606.4257812, 12373.1035156, -29470.0351562, 29955.2851562
3: -5922.4228516, 16098.1152344, -6454.7070312, 17559.9023438, -23482.3242188, 22552.8222656
4: -18788.8964844, 11219.3486328, -20440.9648438, 12221.8476562, -31010.7421875, 31660.3105469

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9675473, upper bound: 19343.0222090
time: 0.95 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7515820, upper bound: 19342.9657431
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16657.4628906, 12336.3632812, -15759.6269531, 11699.0488281, -28356.5117188, 28095.9902344
1: -13484.9306641, 11939.6171875, -12756.6972656, 11328.7353516, -24813.6582031, 24696.3144531
2: -19663.8789062, 13043.1787109, -18606.4257812, 12373.1035156, -32036.9824219, 31649.6035156
3: -6806.5585938, 18534.3945312, -6454.7070312, 17559.9023438, -24366.4609375, 24989.1015625
4: -21611.0488281, 12901.0048828, -20440.9648438, 12221.8476562, -33832.8984375, 33341.9648438

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9675473, upper bound: 19343.0222090
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7515820, upper bound: 19342.9657431
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14321.6904297, 10608.5498047, -17826.6933594, 13224.8486328, -27546.5390625, 28435.2421875
1: -11591.9804688, 10271.3095703, -14433.2353516, 12807.5273438, -24399.5078125, 24704.5429688
2: -16908.6601562, 11221.2412109, -21050.2636719, 13988.1064453, -30896.7617188, 32271.5039062
3: -5855.2421875, 15923.6630859, -7300.1835938, 19878.4121094, -25733.6542969, 23223.8476562
4: -18579.3261719, 11091.2451172, -23127.9726562, 13826.8085938, -32406.1347656, 34219.2187500

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9340307, upper bound: 19342.9477577
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7119877, upper bound: 19342.8951960
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -16267.9394531, 12190.0009766, -17762.5507812, 13184.5312500, -29452.4687500, 29952.5507812
1: -13166.8505859, 11797.3457031, -14380.4755859, 12768.6035156, -25935.4531250, 26177.8203125
2: -19257.9726562, 12870.4697266, -20978.1230469, 13945.6210938, -33203.5937500, 33848.5937500
3: -6676.6625977, 18171.0625000, -7277.0722656, 19814.1777344, -26490.8398438, 25448.1347656
4: -21159.2207031, 12704.6210938, -23048.0800781, 13784.2246094, -34943.4453125, 35752.7031250

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9431481, upper bound: 19342.9986535
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7200704, upper bound: 19342.9183131
time: 2.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15063.8242188, 11140.5283203, -15867.4531250, 11776.0732422, -26839.8984375, 27007.9804688
1: -12192.7910156, 10784.9082031, -12844.0781250, 11403.5126953, -23596.3007812, 23628.9863281
2: -17779.9570312, 11782.0312500, -18733.8574219, 12454.1884766, -30234.1445312, 30515.8847656
3: -6151.5625000, 16738.3984375, -6497.6352539, 17679.6191406, -23831.1796875, 23236.0332031
4: -19540.1679688, 11647.1572266, -20581.5800781, 12302.3242188, -31842.4902344, 32228.7382812

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9658467, upper bound: 19343.0202371
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7498966, upper bound: 19342.9631023
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17152.9394531, 12690.8203125, -15867.4531250, 11776.0732422, -28929.0078125, 28558.2714844
1: -13886.5751953, 12282.2792969, -12844.0781250, 11403.5126953, -25290.0859375, 25126.3554688
2: -20249.8613281, 13416.8525391, -18733.8574219, 12454.1884766, -32704.0468750, 32150.7109375
3: -7008.2280273, 19084.0136719, -6497.6352539, 17679.6191406, -24687.8457031, 25581.6484375
4: -22256.2187500, 13272.7421875, -20581.5800781, 12302.3242188, -34558.5429688, 33854.3164062

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9658467, upper bound: 19343.0202371
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7498966, upper bound: 19342.9631023
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14911.2968750, 11026.2382812, -17929.4980469, 13300.5273438, -28211.8242188, 28955.7363281
1: -12069.8798828, 10677.1835938, -14516.1074219, 12881.0341797, -24950.9140625, 25193.2910156
2: -17604.8515625, 11663.9824219, -21172.2187500, 14068.1601562, -31673.0097656, 32836.2031250
3: -6090.1625977, 16576.8339844, -7342.7792969, 19993.1582031, -26083.3203125, 23919.6113281
4: -19345.2187500, 11528.7548828, -23262.7539062, 13906.0566406, -33251.2734375, 34791.5078125

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9333845, upper bound: 19342.9464340
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7113533, upper bound: 19342.8921177
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16875.7695312, 12625.6972656, -17864.2148438, 13259.5058594, -30135.2734375, 30489.9121094
1: -13658.5986328, 12219.9619141, -14462.4257812, 12841.4296875, -26500.0253906, 26682.3828125
2: -19974.0625000, 13331.5781250, -21098.7812500, 14024.9453125, -33999.0078125, 34430.3593750
3: -6921.1801758, 18843.9960938, -7319.2778320, 19927.7890625, -26848.9687500, 26163.2734375
4: -21947.5859375, 13160.9082031, -23181.4218750, 13862.7412109, -35810.3203125, 36342.3281250

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9414434, upper bound: 19342.9952459
time: 1.01 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7186967, upper bound: 19342.9152769
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15255.0937500, 11337.3496094, -15759.6269531, 11699.0488281, -26954.1406250, 27096.9726562
1: -12348.2080078, 10977.1308594, -12756.6972656, 11328.7353516, -23676.9355469, 23733.8281250
2: -18010.2109375, 11989.8974609, -18606.4257812, 12373.1035156, -30383.3125000, 30596.3203125
3: -6253.8476562, 17000.0175781, -6454.7070312, 17559.9023438, -23813.7480469, 23454.7226562
4: -19785.3125000, 11843.1386719, -20440.9648438, 12221.8476562, -32007.1601562, 32284.1035156

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9594240, upper bound: 19342.7984304
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7474644, upper bound: 19342.7843744
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17397.1445312, 12911.1464844, -15759.6269531, 11699.0488281, -29096.1933594, 28670.7714844
1: -14084.9609375, 12503.2597656, -12756.6972656, 11328.7353516, -25413.6933594, 25259.9570312
2: -20542.0156250, 13656.3964844, -18606.4257812, 12373.1035156, -32915.1171875, 32262.8222656
3: -7125.1450195, 19397.3964844, -6454.7070312, 17559.9023438, -24685.0468750, 25852.1035156
4: -22568.6523438, 13498.4541016, -20440.9648438, 12221.8476562, -34790.4960938, 33939.4179688

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9594240, upper bound: 19342.8060338
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7474644, upper bound: 19342.7925204
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14929.6875000, 11099.9179688, -17668.9980469, 13108.9492188, -28038.6367188, 28768.9160156
1: -12085.0156250, 10747.1035156, -14305.7988281, 12695.1367188, -24780.1503906, 25052.8984375
2: -17623.0000000, 11740.6933594, -20862.8359375, 13865.5253906, -31488.5253906, 32603.5234375
3: -6122.3735352, 16635.3027344, -7235.9853516, 19701.4921875, -25823.8652344, 23871.2890625
4: -19360.4355469, 11595.5937500, -22922.3671875, 13705.6562500, -33066.0898438, 34517.9609375

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535601, upper bound: 19342.9419163
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7535601, upper bound: 19342.9419163
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15304.0664062, 11376.9824219, -17751.5058594, 13169.0937500, -28473.1601562, 29128.4882812
1: -12387.4824219, 11014.5996094, -14372.4375000, 12753.6484375, -25141.1289062, 25387.0351562
2: -18069.0976562, 12031.1328125, -20961.3671875, 13929.2216797, -31998.3203125, 32992.5000000
3: -6276.4975586, 17056.5156250, -7269.1147461, 19794.3339844, -26070.8320312, 24325.6289062
4: -19849.8808594, 11882.6572266, -23030.2832031, 13768.5439453, -33618.4257812, 34912.9335938

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7482132, upper bound: 19342.7893500
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7482132, upper bound: 19342.7893500
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -15808.7578125, 11732.4414062, -15867.4531250, 11776.0732422, -27584.8320312, 27599.8945312
1: -12796.6992188, 11361.1044922, -12844.0781250, 11403.5126953, -24200.2089844, 24205.1816406
2: -18665.0468750, 12407.6552734, -18733.8574219, 12454.1884766, -31119.2343750, 31141.5117188
3: -6472.9467773, 17614.5253906, -6497.6352539, 17679.6191406, -24152.5644531, 24112.1601562
4: -20505.8945312, 12256.4169922, -20581.5800781, 12302.3242188, -32808.2187500, 32837.9960938

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9449851, upper bound: 19342.7561046
time: 1.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7398033, upper bound: 19342.7433511
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -17867.7812500, 13254.6699219, -15867.4531250, 11776.0732422, -29643.8554688, 29122.1230469
1: -14466.2548828, 12836.6572266, -12844.0781250, 11403.5126953, -25869.7617188, 25680.7343750
2: -21099.7460938, 14019.5605469, -18733.8574219, 12454.1884766, -33553.9335938, 32753.4179688
3: -7317.1787109, 19924.7968750, -6497.6352539, 17679.6191406, -24996.7968750, 26422.4316406
4: -23183.0937500, 13857.9882812, -20581.5800781, 12302.3242188, -35485.4179688, 34439.5625000

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.9449851, upper bound: 19342.7567920
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7398033, upper bound: 19342.7449502
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -15504.6279297, 11512.0810547, -17771.3496094, 13184.2880859, -28688.9160156, 29283.4277344
1: -12550.7431641, 11147.4677734, -14388.3271484, 12768.3291016, -25319.0722656, 25535.7949219
2: -18302.7792969, 12176.4746094, -20984.2949219, 13945.2070312, -32247.9824219, 33160.7617188
3: -6351.2114258, 17273.7890625, -7278.3608398, 19815.7714844, -26166.9785156, 24552.1464844
4: -20108.4980469, 12026.6308594, -23056.5839844, 13784.5195312, -33893.0156250, 35083.2109375

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7521887, upper bound: 19342.9363087
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7521887, upper bound: 19342.9363087
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -15841.5400391, 11758.2138672, -17852.6250000, 13243.5751953, -29085.1152344, 29610.8339844
1: -12822.6953125, 11386.1210938, -14453.9589844, 12825.9814453, -25648.6757812, 25840.0800781
2: -18704.2441406, 12434.8437500, -21081.2968750, 14008.0097656, -32712.2500000, 33516.1406250
3: -6488.5131836, 17652.8593750, -7311.0590820, 19907.2226562, -26395.7363281, 24963.9140625
4: -20548.7187500, 12282.1689453, -23162.8281250, 13846.5429688, -34395.2617188, 35444.9960938

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7417055, upper bound: 19342.7417055
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7417055, upper bound: 19342.7417055
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -15255.0937500, 11337.3496094, -18302.8144531, 13523.4892578, -28778.5781250, 29640.1640625
1: -12348.2080078, 10977.1308594, -14815.7207031, 13093.6181641, -25441.8261719, 25792.8515625
2: -18010.2109375, 11989.8974609, -21620.8496094, 14301.4716797, -32311.6796875, 33610.7421875
3: -6253.8476562, 17000.0175781, -7464.0634766, 20382.9492188, -26636.7949219, 24464.0800781
4: -19785.3125000, 11843.1386719, -23744.8593750, 14122.0585938, -33907.3710938, 35588.0000000

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8225282, upper bound: 19342.6036956
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8518136, upper bound: 19342.9080264
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8697006, upper bound: 19342.9045063
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -17397.1445312, 12911.1464844, -18302.8144531, 13523.4892578, -30920.6289062, 31213.9609375
1: -14084.9609375, 12503.2597656, -14815.7207031, 13093.6181641, -27178.5781250, 27318.9804688
2: -20542.0156250, 13656.3964844, -21620.8496094, 14301.4716797, -34843.4843750, 35277.2343750
3: -7125.1450195, 19397.3964844, -7464.0634766, 20382.9492188, -27508.0937500, 26861.4609375
4: -22568.6523438, 13498.4541016, -23744.8593750, 14122.0585938, -36690.7031250, 37243.3125000

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8225282, upper bound: 19342.6075256
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8518136, upper bound: 19342.9080265
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8697006, upper bound: 19342.9045064
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -15093.8017578, 11219.2373047, -20303.7031250, 14990.1279297, -30083.9296875, 31522.9394531
1: -12218.8242188, 10864.7246094, -16438.7402344, 14516.9736328, -26735.7968750, 27303.4628906
2: -17825.3417969, 11863.5791016, -23985.5957031, 15857.1152344, -33682.4492188, 35849.1757812
3: -6188.0126953, 16828.7031250, -8279.7968750, 22621.1582031, -28809.1718750, 25108.5000000
4: -19579.4296875, 11716.3427734, -26344.9746094, 15668.5576172, -35247.9882812, 38061.3164062

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7827155, upper bound: 19342.8107020
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7827155, upper bound: 19342.8107020
time: 0.96 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.38 seconds
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9222212, upper bound: 19342.9652583
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19343.0205696, upper bound: 19342.9681771
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9222212, upper bound: 19342.9652583
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19343.0205696, upper bound: 19342.9681771
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19343.0099132, upper bound: 19342.9262134
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.6696446, upper bound: 19342.8085455
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.8735318, upper bound: 19342.6832340
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.6429665, upper bound: 19342.5841148
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9199390, upper bound: 19342.9606962
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19343.0173755, upper bound: 19342.9637199
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9199390, upper bound: 19342.9606962
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19343.0173755, upper bound: 19342.9637199
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9464340, upper bound: 19342.9333845
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9952459, upper bound: 19342.9414434
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.8193656, upper bound: 19342.6816901
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.6017187, upper bound: 19342.5830691
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9675473, upper bound: 19343.0222090
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7515820, upper bound: 19342.9657431
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9675473, upper bound: 19343.0222090
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7515820, upper bound: 19342.9657431
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9340307, upper bound: 19342.9477577
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7119877, upper bound: 19342.8951960
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9431481, upper bound: 19342.9986535
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7200704, upper bound: 19342.9183131
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9658467, upper bound: 19343.0202371
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7498966, upper bound: 19342.9631023
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9658467, upper bound: 19343.0202371
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7498966, upper bound: 19342.9631023
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9333845, upper bound: 19342.9464340
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7113533, upper bound: 19342.8921177
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9414434, upper bound: 19342.9952459
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7186967, upper bound: 19342.9152769
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9594240, upper bound: 19342.7984304
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7474644, upper bound: 19342.7843744
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9594240, upper bound: 19342.8060338
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7474644, upper bound: 19342.7925204
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7535601, upper bound: 19342.9419163
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7535601, upper bound: 19342.9419163
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7482132, upper bound: 19342.7893500
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7482132, upper bound: 19342.7893500
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9449851, upper bound: 19342.7561046
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7398033, upper bound: 19342.7433511
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.9449851, upper bound: 19342.7567920
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7398033, upper bound: 19342.7449502
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7521887, upper bound: 19342.9363087
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7521887, upper bound: 19342.9363087
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7417055, upper bound: 19342.7417055
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7417055, upper bound: 19342.7417055
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.8518136, upper bound: 19342.9080264
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.8697006, upper bound: 19342.9045063
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.8518136, upper bound: 19342.9080265
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.8697006, upper bound: 19342.9045064
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7827155, upper bound: 19342.8107020
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.38
Output dim: 3, lower bound: -19342.7827155, upper bound: 19342.8107020
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8282612, upper bound: 19342.8661675
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8432017, upper bound: 19342.8843922
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8119476, upper bound: 19342.8045930
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8253847, upper bound: 19342.8571131
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8724994, upper bound: 19342.8695315
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8721501
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8724994, upper bound: 19342.8695315
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8725057, upper bound: 19342.8721501
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7901639, upper bound: 19342.8449697
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7851911, upper bound: 19342.8102888
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7901639, upper bound: 19342.8540992
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7851911, upper bound: 19342.8271444
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7974873
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7999659
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7974873
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.8672970, upper bound: 19342.7999659
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7856997, upper bound: 19342.7849397
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7819080, upper bound: 19342.7732374
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7856997, upper bound: 19342.7864396
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.38
Output dim: 3, lower bound: -19342.7819080, upper bound: 19342.7756401
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.2500000, mid=0.2500000, abs_max=24289.171875
rel_dist={3: [-19343.120444796758, 19343.120444796754]}

## Binary search (step 2) starts
Candidate diff: 0.1250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8803278, upper bound: 19343.0129860
time: 0.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8766225, upper bound: 19342.8766225
time: 0.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 3, lower bound: -19342.8803278, upper bound: 19343.0129860
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.75
Output dim: 3, lower bound: -19342.8766225, upper bound: 19342.8766225

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -15606.2392578, 11546.4335938, -15777.2783203, 11673.5341797, -27279.7734375, 27323.7109375
1: -12631.3779297, 11177.4941406, -12769.6708984, 11299.6572266, -23931.0332031, 23947.1640625
2: -18419.4785156, 12209.6884766, -18622.0156250, 12343.2011719, -30762.6796875, 30831.7031250
3: -6374.2680664, 17344.5468750, -6443.6108398, 17536.1230469, -23910.3886719, 23788.1542969
4: -20242.6933594, 12071.0019531, -20465.1875000, 12203.7958984, -32446.4843750, 32536.1894531

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4634957, upper bound: 19342.6157165
time: 0.83 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8647942, upper bound: 19342.9479705
time: 0.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -17825.7265625, 13106.9628906, -15629.4570312, 11560.5791016, -29386.3046875, 28736.4179688
1: -14429.7958984, 12691.0000000, -12650.1250000, 11189.4736328, -25619.2675781, 25341.1250000
2: -21053.4453125, 13860.3105469, -18449.1035156, 12222.5087891, -33275.9492188, 32309.4140625
3: -7247.8081055, 19801.2207031, -6380.6689453, 17371.0410156, -24618.8476562, 26181.8886719
4: -23130.5800781, 13698.3564453, -20274.9140625, 12084.4824219, -35215.0585938, 33973.2695312

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19342.4618716, upper bound: 19342.5780917
time: 1.00 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8585390, upper bound: 19342.8585390
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.55 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 3.55
Output dim: 3, lower bound: -19342.4634957, upper bound: 19342.6157165
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -19342.8647942, upper bound: 19342.9479705
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 3.55
Output dim: 3, lower bound: -19342.4618716, upper bound: 19342.5780917
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -19342.8585390, upper bound: 19342.8585390

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -15343.4853516, 11355.8740234, -16286.5214844, 12087.1660156, -27430.6523438, 27642.3945312
1: -12418.0839844, 10994.0546875, -13183.0224609, 11704.1044922, -24122.1816406, 24177.0781250
2: -18112.4902344, 12010.3730469, -19229.8671875, 12783.2705078, -30895.7597656, 31240.2402344
3: -6269.6762695, 17056.7343750, -6670.0820312, 18148.3027344, -24417.9746094, 23726.8164062
4: -19904.3066406, 11872.6943359, -21126.1406250, 12627.5947266, -32531.9003906, 32998.8320312

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7879516, upper bound: 19342.9429296
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8645475, upper bound: 19342.9478846
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8647942, upper bound: 19342.9478846
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -17577.7753906, 12927.9160156, -16108.0937500, 11953.9863281, -29531.7578125, 29036.0097656
1: -14228.9648438, 12518.9228516, -13039.0917969, 11574.0087891, -25802.9726562, 25558.0117188
2: -20763.9394531, 13673.0029297, -19021.5644531, 12640.6445312, -33404.5820312, 32694.5644531
3: -7148.3081055, 19529.1347656, -6594.8417969, 17949.9511719, -25098.2578125, 26123.9765625
4: -22810.8496094, 13512.0957031, -20896.5175781, 12486.4482422, -35297.2968750, 34408.6093750

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7849036, upper bound: 19342.8530599
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7802033, upper bound: 19342.7802033
time: 0.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.58 seconds
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 3, lower bound: -19342.8645475, upper bound: 19342.9478846
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 3, lower bound: -19342.8647942, upper bound: 19342.9478846
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 3, lower bound: -19342.7849036, upper bound: 19342.8530599
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.58
Output dim: 3, lower bound: -19342.7802033, upper bound: 19342.7802033

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14672.6240234, 10877.2285156, -16036.8466797, 11907.5380859, -26580.1582031, 26914.0742188
1: -11874.6044922, 10528.9931641, -12980.9228516, 11529.5693359, -23404.1738281, 23509.9160156
2: -17321.0000000, 11503.6015625, -18935.4804688, 12593.5732422, -29914.5742188, 30439.0800781
3: -6001.8593750, 16313.6572266, -6569.6928711, 17871.5703125, -23873.4257812, 22883.3496094
4: -19033.6035156, 11371.2255859, -20801.4277344, 12439.5654297, -31473.1679688, 32172.6523438

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8546240, upper bound: 19342.9416594
time: 2.91 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8358801, upper bound: 19342.9367794
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -15245.2050781, 11281.4003906, -16228.7949219, 12044.4228516, -27289.6289062, 27510.1953125
1: -12338.8681641, 10922.0781250, -13136.4208984, 11662.5312500, -24001.3945312, 24058.4980469
2: -17997.1367188, 11931.9375000, -19162.1582031, 12737.7373047, -30734.8750000, 31094.0957031
3: -6228.9018555, 16947.4160156, -6645.9682617, 18084.4003906, -24313.3027344, 23593.3847656
4: -19777.3886719, 11794.3964844, -21051.6523438, 12582.6914062, -32360.0761719, 32846.0468750

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8543100, upper bound: 19342.9372423
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8347493, upper bound: 19342.9319441
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -17312.5292969, 12727.3466797, -15934.9257812, 11827.5468750, -29140.0761719, 28662.2734375
1: -14014.1679688, 12325.6093750, -12898.8037109, 11451.8632812, -25466.0312500, 25224.4140625
2: -20450.4843750, 13459.9277344, -18816.9648438, 12506.7939453, -32957.2773438, 32276.8925781
3: -7040.1870117, 19233.3867188, -6524.3618164, 17757.7832031, -24797.9707031, 25757.7480469
4: -22466.7500000, 13300.3710938, -20671.9023438, 12353.8037109, -34820.5546875, 33972.2734375

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7826340, upper bound: 19342.8526015
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7849036, upper bound: 19342.8530599
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -18873.4218750, 13831.2675781, -15717.8125000, 11657.2558594, -30530.6777344, 29549.0742188
1: -15275.7382812, 13399.4755859, -12724.6953125, 11287.9335938, -26563.6699219, 26124.1718750
2: -22271.6308594, 14630.2685547, -18560.5742188, 12327.8701172, -34599.4960938, 33190.8437500
3: -7641.7968750, 20948.6132812, -6430.2524414, 17509.8457031, -25151.6406250, 27378.8652344
4: -24458.8750000, 14450.4550781, -20387.0292969, 12175.7207031, -36634.5937500, 34837.4843750

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7778422, upper bound: 19342.7801422
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7802033, upper bound: 19342.7802033
time: 0.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.71 seconds
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.8546240, upper bound: 19342.9416594
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.8358801, upper bound: 19342.9367794
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.8543100, upper bound: 19342.9372423
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.8347493, upper bound: 19342.9319441
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.7826340, upper bound: 19342.8526015
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.7849036, upper bound: 19342.8530599
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.7778422, upper bound: 19342.7801422
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.71
Output dim: 3, lower bound: -19342.7802033, upper bound: 19342.7802033

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14592.9042969, 10817.4902344, -15810.2001953, 11738.1123047, -26331.0156250, 26627.6914062
1: -11810.1396484, 10471.0869141, -12797.6337891, 11365.8730469, -23176.0097656, 23268.7128906
2: -17226.6972656, 11440.3457031, -18667.3476562, 12414.6113281, -29641.3085938, 30107.6894531
3: -5968.9667969, 16224.2089844, -6475.9150391, 17617.5195312, -23586.4824219, 22700.1250000
4: -18930.0722656, 11308.8535156, -20507.1347656, 12263.0517578, -31193.1250000, 31815.9882812

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7846655, upper bound: 19342.9367573
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8266764, upper bound: 19342.8747421
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14526.9726562, 10767.8955078, -17890.2949219, 13271.0283203, -27797.9960938, 28658.1894531
1: -11756.7939453, 10423.3261719, -14484.6025391, 12851.3925781, -24608.1835938, 24907.9218750
2: -17149.3359375, 11388.0185547, -21126.5957031, 14036.6796875, -31186.0156250, 32514.6132812
3: -5940.8134766, 16150.8701172, -7325.0922852, 19949.1914062, -25890.0039062, 23475.9589844
4: -18844.9160156, 11256.9101562, -23210.9765625, 13875.0947266, -32720.0117188, 34467.8867188

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7753110, upper bound: 19342.9290465
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8091498, upper bound: 19342.8697570
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8186004, upper bound: 19342.9198033
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15165.8916016, 11221.8242188, -16002.5693359, 11875.3261719, -27041.2187500, 27224.3945312
1: -12274.7548828, 10864.5312500, -12953.5068359, 11499.1123047, -23773.8671875, 23818.0371094
2: -17903.3671875, 11869.1816406, -18894.6289062, 12559.0751953, -30462.4375000, 30763.8105469
3: -6196.0981445, 16858.4667969, -6552.3408203, 17830.9609375, -24027.0585938, 23410.8085938
4: -19674.4355469, 11732.1806641, -20757.9511719, 12406.4355469, -32080.8671875, 32490.1328125

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7833019, upper bound: 19342.9324870
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8264159, upper bound: 19342.8716074
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -15115.8798828, 11184.7529297, -18064.6054688, 13400.2138672, -28516.0937500, 29249.3593750
1: -12234.2666016, 10828.7089844, -14625.5488281, 12976.7353516, -25211.0000000, 25454.2539062
2: -17844.7695312, 11830.2099609, -21333.1347656, 14173.0976562, -32017.8613281, 33163.3398438
3: -6175.3818359, 16803.1660156, -7397.5849609, 20144.6953125, -26320.0761719, 24200.7460938
4: -19609.9492188, 11693.3564453, -23439.1992188, 14010.1250000, -33620.0742188, 35132.5546875

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7738675, upper bound: 19342.9253806
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8086080, upper bound: 19342.8665735
time: 1.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8173118, upper bound: 19342.9168187
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17080.0253906, 12564.4111328, -15286.6640625, 11362.8056641, -28442.8261719, 27851.0742188
1: -13825.7333984, 12166.9794922, -12374.2753906, 10999.5185547, -24825.2480469, 24541.2539062
2: -20176.3183594, 13287.4101562, -18051.8183594, 12014.2675781, -32190.5839844, 31339.2285156
3: -6947.3339844, 18977.1054688, -6265.6923828, 17038.8085938, -23986.1425781, 25242.7968750
4: -22164.4746094, 13129.1425781, -19829.8203125, 11866.6289062, -34031.1015625, 32958.9570312

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8238157
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17240.9765625, 12672.9023438, -15841.1552734, 11757.8945312, -28998.8671875, 28514.0566406
1: -13956.4326172, 12272.4277344, -12823.0917969, 11384.3662109, -25340.7988281, 25095.5195312
2: -20366.3320312, 13402.0947266, -18706.9570312, 12432.5673828, -32798.8984375, 32109.0507812
3: -7010.3891602, 19152.9941406, -6485.0625000, 17653.8769531, -24664.2636719, 25638.0566406
4: -22374.2109375, 13243.3281250, -20550.9121094, 12280.5976562, -34654.8085938, 33794.2421875

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8235333
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -18669.8515625, 13684.8349609, -15062.0781250, 11187.3808594, -29857.2285156, 28746.9101562
1: -15111.4091797, 13256.3203125, -12194.0947266, 10830.9326172, -25942.3359375, 25450.4140625
2: -22031.6093750, 14474.4521484, -17786.7031250, 11830.5019531, -33862.1054688, 32261.1542969
3: -7558.0747070, 20722.0839844, -6168.7802734, 16783.0371094, -24341.1113281, 26890.8632812
4: -24193.5156250, 14295.6787109, -19535.4023438, 11683.5703125, -35877.0820312, 33831.0781250

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -18792.4570312, 13772.1630859, -15624.1953125, 11587.6503906, -30380.1074219, 29396.3554688
1: -15210.5224609, 13341.7871094, -12649.1445312, 11220.4853516, -26431.0019531, 25990.9218750
2: -22178.4882812, 14567.4931641, -18450.8417969, 12253.6748047, -34432.1562500, 33018.3359375
3: -7609.3393555, 20859.9824219, -6390.9716797, 17406.1816406, -25015.5175781, 27250.9531250
4: -24356.6132812, 14388.7666016, -20266.3046875, 12102.5332031, -36459.1445312, 34655.0664062

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848
time: 0.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.71 seconds
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8266764, upper bound: 19342.8747421
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8091498, upper bound: 19342.8697570
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8186004, upper bound: 19342.9198033
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8264159, upper bound: 19342.8716074
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8086080, upper bound: 19342.8665735
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.8173118, upper bound: 19342.9168187
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8238157
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8235333
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.71
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -14292.1660156, 10591.1005859, -15810.2001953, 11738.1123047, -26030.2773438, 26401.2988281
1: -11567.6699219, 10255.4941406, -12797.6337891, 11365.8730469, -22933.5390625, 23053.1230469
2: -16876.3222656, 11204.3925781, -18667.3476562, 12414.6113281, -29290.9335938, 29871.7382812
3: -5845.9492188, 15895.8095703, -6475.9150391, 17617.5195312, -23463.4648438, 22371.7246094
4: -18542.7382812, 11074.0341797, -20507.1347656, 12263.0517578, -30805.7890625, 31581.1679688

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8266764, upper bound: 19342.8747421
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8266764, upper bound: 19342.8747421
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -16222.6259766, 12158.5136719, -15684.8535156, 11656.2822266, -27878.9082031, 27843.3652344
1: -13130.5439453, 11768.0556641, -12695.0468750, 11286.9130859, -24417.4531250, 24463.1015625
2: -19206.4882812, 12838.8691406, -18524.5507812, 12328.8056641, -31535.2871094, 31363.4179688
3: -6659.7636719, 18123.3476562, -6429.3457031, 17488.6406250, -24148.4023438, 24552.6933594
4: -21101.1289062, 12672.3134766, -20350.4863281, 12177.1748047, -33278.3046875, 33022.7968750

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -14225.8017578, 10541.0761719, -17890.2949219, 13271.0283203, -27496.8281250, 28431.3710938
1: -11513.9208984, 10207.3203125, -14484.6025391, 12851.3925781, -24365.3125000, 24691.9179688
2: -16798.3593750, 11151.6992188, -21126.5957031, 14036.6796875, -30835.0371094, 32278.2949219
3: -5817.5952148, 15821.9121094, -7325.0922852, 19949.1914062, -25766.7851562, 23147.0039062
4: -18456.9257812, 11021.6367188, -23210.9765625, 13875.0947266, -32332.0195312, 34232.6132812

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8091498, upper bound: 19342.8697570
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8091498, upper bound: 19342.8697570
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -16165.0556641, 12114.7470703, -17765.2304688, 13192.7197266, -29357.7753906, 29879.9746094
1: -13083.9335938, 11726.0810547, -14381.8496094, 12775.7636719, -25859.6972656, 26107.9296875
2: -19138.9648438, 12792.8193359, -20986.0644531, 13954.2851562, -33093.2500000, 33778.8828125
3: -6634.8359375, 18059.1308594, -7280.7485352, 19823.8007812, -26458.6367188, 25339.8789062
4: -21026.6679688, 12626.4082031, -23055.2597656, 13792.6123047, -34819.2812500, 35681.6679688

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8186004, upper bound: 19342.9198033
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8186004, upper bound: 19342.9198033
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -14864.4892578, 10995.2011719, -16002.5693359, 11875.3261719, -26739.8164062, 26997.7695312
1: -12031.5576172, 10648.3916016, -12953.5068359, 11499.1123047, -23530.6699219, 23601.8984375
2: -17551.9765625, 11632.7685547, -18894.6289062, 12559.0751953, -30111.0449219, 30527.3984375
3: -6072.9863281, 16529.3085938, -6552.3408203, 17830.9609375, -23903.9453125, 23081.6484375
4: -19286.0390625, 11497.1298828, -20757.9511719, 12406.4355469, -31692.4726562, 32255.0742188

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8264159, upper bound: 19342.8716074
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8264159, upper bound: 19342.8716074
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -16812.8476562, 12580.3085938, -15874.4970703, 11791.8027344, -28604.6484375, 28454.8046875
1: -13607.9619141, 12177.2802734, -12848.6474609, 11418.5234375, -25026.4804688, 25025.9277344
2: -19901.6875000, 13285.3291016, -18748.8632812, 12471.4687500, -32373.1542969, 32034.1914062
3: -6896.1435547, 18776.3144531, -6504.8291016, 17699.3925781, -24595.5351562, 25281.1445312
4: -21866.5410156, 13114.0878906, -20598.0312500, 12318.7890625, -34185.3281250, 33712.1171875

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -14814.0175781, 10957.7373047, -18064.6054688, 13400.2138672, -28214.2304688, 29022.3417969
1: -11990.6953125, 10612.1904297, -14625.5488281, 12976.7353516, -24967.4296875, 25237.7382812
2: -17492.8339844, 11593.3730469, -21333.1347656, 14173.0976562, -31665.9257812, 32926.5039062
3: -6052.0712891, 16473.4824219, -7397.5849609, 20144.6953125, -26196.7636719, 23871.0644531
4: -19220.9550781, 11458.0722656, -23439.1992188, 14010.1250000, -33231.0781250, 34897.2734375

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8086080, upper bound: 19342.8665735
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8086080, upper bound: 19342.8665735
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -16769.9335938, 12548.1259766, -17936.8339844, 13320.2548828, -30090.1875000, 30484.9589844
1: -13573.2539062, 12146.4414062, -14520.5048828, 12899.5029297, -26472.7578125, 26666.9453125
2: -19851.6152344, 13251.6005859, -21189.5058594, 14088.9658203, -33940.5820312, 34441.1054688
3: -6878.2021484, 18728.8554688, -7352.3037109, 20016.6152344, -26894.8164062, 26081.1601562
4: -21811.3222656, 13080.4687500, -23280.0605469, 13925.8837891, -35737.2070312, 36360.5312500

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8173118, upper bound: 19342.9168187
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8173118, upper bound: 19342.9168187
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -16857.2656250, 12396.5947266, -15207.8349609, 11303.8535156, -28161.1191406, 27604.4238281
1: -13645.6601562, 12005.1806641, -12310.5029297, 10942.6533203, -24588.3125000, 24315.6835938
2: -19913.0097656, 13110.3173828, -17958.4863281, 11952.1562500, -31865.1640625, 31068.8046875
3: -6855.3081055, 18727.6835938, -6233.1596680, 16950.4375000, -23805.7421875, 24960.8437500
4: -21875.3027344, 12953.9511719, -19727.3750000, 11805.3505859, -33680.6445312, 32681.3222656

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -18910.3105469, 13909.8837891, -15147.7011719, 11258.6445312, -30168.9550781, 29057.5859375
1: -15310.9765625, 13469.1982422, -12261.9599609, 10898.9960938, -26209.9707031, 25731.1562500
2: -22338.8164062, 14708.9052734, -17888.0742188, 11904.1083984, -34242.9179688, 32596.9804688
3: -7690.5825195, 21028.7187500, -6207.1411133, 16883.6660156, -24574.2460938, 27235.8593750
4: -24542.2343750, 14542.1298828, -19649.8300781, 11757.7060547, -36299.9414062, 34191.9570312

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8238157
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8238157
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -17024.1152344, 12509.9833984, -15761.4580078, 11698.2910156, -28722.4042969, 28271.4414062
1: -13781.0664062, 12115.2919922, -12758.6464844, 11326.7001953, -25107.7656250, 24873.9355469
2: -20109.8281250, 13230.3906250, -18612.6953125, 12369.6005859, -32479.4277344, 31843.0859375
3: -6920.8989258, 18910.1015625, -6452.0751953, 17564.5585938, -24485.4570312, 25362.1718750
4: -22092.6113281, 13073.0644531, -20447.4355469, 12218.4560547, -34311.0585938, 33520.5000000

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -19078.7167969, 14029.7988281, -15713.8662109, 11663.1894531, -30741.9062500, 29743.6640625
1: -15447.2617188, 13584.5312500, -12720.2041016, 11292.8300781, -26740.0917969, 26304.7343750
2: -22538.5058594, 14834.5244141, -18556.9863281, 12332.4111328, -34870.9179688, 33391.5039062
3: -7760.8652344, 21215.3808594, -6432.2573242, 17512.0371094, -25272.9023438, 27647.6347656
4: -24762.8320312, 14667.9560547, -20386.0957031, 12181.5039062, -36944.3359375, 35054.0468750

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8235333
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8235333
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -18445.4355469, 13515.5917969, -14983.3750000, 11128.3906250, -29573.8261719, 28498.9667969
1: -14929.9404297, 13093.2675781, -12130.4277344, 10773.8681641, -25703.8046875, 25223.6933594
2: -21766.1445312, 14296.1552734, -17693.4785156, 11768.2031250, -33534.3476562, 31989.6328125
3: -7465.0380859, 20470.2324219, -6136.2451172, 16694.7519531, -24159.7890625, 26606.4765625
4: -23901.9394531, 14119.3701172, -19433.0917969, 11622.0898438, -35524.0195312, 33552.4609375

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -20392.9804688, 14954.5732422, -14920.5058594, 11080.9306641, -31473.9101562, 29875.0781250
1: -16509.4179688, 14486.8330078, -12079.6464844, 10728.1884766, -27237.6054688, 26566.4804688
2: -24068.1914062, 15818.2099609, -17619.8027344, 11717.8105469, -35786.0000000, 33438.0078125
3: -8260.7714844, 22652.5878906, -6109.0078125, 16624.7695312, -24885.5410156, 28761.5957031
4: -26432.3847656, 15631.2373047, -19351.9160156, 11572.1093750, -38004.4921875, 34983.1523438

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -18572.6718750, 13606.8427734, -15544.2558594, 11527.7187500, -30100.3886719, 29151.0976562
1: -15032.9199219, 13182.1328125, -12584.5156250, 11162.3857422, -26195.3027344, 25766.6484375
2: -21918.6347656, 14392.8427734, -18356.2597656, 12190.2060547, -34108.8398438, 32749.1015625
3: -7518.2690430, 20613.2851562, -6357.8129883, 17316.5253906, -24834.7910156, 26971.0976562
4: -24071.2128906, 14216.3876953, -20162.4628906, 12039.8798828, -36111.0937500, 34378.8515625

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -20544.3730469, 15064.2939453, -15495.9990234, 11492.1484375, -32036.5214844, 30560.2910156
1: -16631.5820312, 14592.1855469, -12545.5097656, 11128.1689453, -27759.7500000, 27137.6933594
2: -24247.5625000, 15932.8671875, -18299.7207031, 12152.6259766, -36400.1875000, 34232.5859375
3: -8324.9707031, 22821.5292969, -6337.7382812, 17263.2089844, -25588.1738281, 29159.2656250
4: -26630.6738281, 15746.4394531, -20100.2011719, 12002.5312500, -38633.2031250, 35846.6328125

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.65 seconds
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8266764, upper bound: 19342.8747421
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8266764, upper bound: 19342.8747421
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8091498, upper bound: 19342.8697570
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8091498, upper bound: 19342.8697570
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8186004, upper bound: 19342.9198033
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8186004, upper bound: 19342.9198033
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8264159, upper bound: 19342.8716074
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8264159, upper bound: 19342.8716074
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8086080, upper bound: 19342.8665735
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8086080, upper bound: 19342.8665735
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8173118, upper bound: 19342.9168187
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.8173118, upper bound: 19342.9168187
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8238157
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8238157
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7739958, upper bound: 19342.8424422
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8235333
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7699543, upper bound: 19342.8235333
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14292.1660156, 10591.1005859, -15652.0078125, 11621.6513672, -25913.8164062, 26243.1074219
1: -11567.6699219, 10255.4941406, -12669.5849609, 11253.5400391, -22821.2070312, 22925.0781250
2: -16876.3222656, 11204.3925781, -18479.5175781, 12291.3769531, -29167.6992188, 29683.9101562
3: -5845.9492188, 15895.8095703, -6411.4755859, 17440.6113281, -23286.5585938, 22307.2851562
4: -18542.7382812, 11074.0341797, -20300.9804688, 12140.8427734, -30683.5742188, 31375.0117188

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697455, upper bound: 19342.8510653
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7691589, upper bound: 19342.7928483
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14292.1660156, 10591.1005859, -18201.3320312, 13451.3193359, -27743.4843750, 28792.4335938
1: -11567.6699219, 10255.4941406, -14733.3154297, 13023.4980469, -24591.1679688, 24988.8085938
2: -16876.3222656, 11204.3925781, -21500.8769531, 14225.3837891, -31101.7070312, 32705.2695312
3: -5845.9492188, 15895.8095703, -7423.2509766, 20270.6230469, -26116.5683594, 23319.0605469
4: -18542.7382812, 11074.0341797, -23612.4843750, 14046.2490234, -32588.9804688, 34686.5156250

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7697455, upper bound: 19342.8510653
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7691589, upper bound: 19342.7928483
time: 0.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16222.6259766, 12158.5136719, -15529.3496094, 11541.8789062, -27764.5039062, 27687.8593750
1: -13130.5439453, 11768.0556641, -12569.1396484, 11176.5781250, -24307.1191406, 24337.1953125
2: -19206.4882812, 12838.8691406, -18339.9414062, 12207.7373047, -31414.2265625, 31178.8085938
3: -6659.7636719, 18123.3476562, -6366.0478516, 17314.7597656, -23974.5195312, 24489.3945312
4: -21101.1289062, 12672.3134766, -20147.8671875, 12057.1367188, -33158.2656250, 32820.1796875

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16222.6259766, 12158.5136719, -18080.2851562, 13372.4990234, -29595.1250000, 30238.7968750
1: -13130.5439453, 11768.0556641, -14634.2138672, 12947.2187500, -26077.7617188, 26402.2695312
2: -19206.4882812, 12838.8691406, -21362.7265625, 14142.5791016, -33349.0664062, 34201.5898438
3: -6659.7636719, 18123.3476562, -7378.3759766, 20145.7636719, -26805.5253906, 25501.7226562
4: -21101.1289062, 12672.3134766, -23460.9355469, 13963.2802734, -35064.4101562, 36133.2421875

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14225.8017578, 10541.0761719, -17729.7109375, 13152.4003906, -27378.2031250, 28270.7871094
1: -11513.9208984, 10207.3203125, -14354.6123047, 12737.3466797, -24251.2656250, 24561.9335938
2: -16798.3593750, 11151.6992188, -20935.8203125, 13911.6376953, -30709.9960938, 32087.5195312
3: -5817.5952148, 15821.9121094, -7259.6791992, 19769.4785156, -25587.0722656, 23081.5917969
4: -18456.9257812, 11021.6367188, -23001.6796875, 13751.1494141, -32208.0742188, 34023.3164062

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7575661, upper bound: 19342.8325564
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7575726, upper bound: 19342.7883602
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14225.8017578, 10541.0761719, -20203.8222656, 14915.5000000, -29141.3007812, 30744.8984375
1: -11513.9208984, 10207.3203125, -16357.8066406, 14444.6689453, -25958.5898438, 26565.1250000
2: -16798.3593750, 11151.6992188, -23867.4609375, 15778.3271484, -32576.6855469, 35019.1601562
3: -5817.5952148, 15821.9121094, -8237.9765625, 22508.8828125, -28326.4785156, 24059.8886719
4: -18456.9257812, 11021.6367188, -26214.8144531, 15590.5185547, -34047.4414062, 37236.4531250

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7575661, upper bound: 19342.8325564
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7575726, upper bound: 19342.7883602
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16165.0556641, 12114.7470703, -17606.2089844, 13075.3291016, -29240.3847656, 29720.9531250
1: -13083.9335938, 11726.0810547, -14253.0664062, 12662.9179688, -25746.8515625, 25979.1484375
2: -19138.9648438, 12792.8193359, -20797.0644531, 13830.5498047, -32969.5156250, 33589.8789062
3: -6634.8359375, 18059.1308594, -7216.0224609, 19645.8398438, -26280.6757812, 25275.1523438
4: -21026.6679688, 12626.4082031, -22847.9277344, 13669.9375000, -34696.6054688, 35474.3359375

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7975584, upper bound: 19342.8141399
time: 0.92 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8158441, upper bound: 19342.9181034
time: 0.96 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16165.0556641, 12114.7470703, -20082.8457031, 14839.3710938, -31004.4257812, 32197.5898438
1: -13083.9335938, 11726.0810547, -16258.5380859, 14371.1230469, -27455.0566406, 27984.6171875
2: -19138.9648438, 12792.8193359, -23731.7167969, 15698.1806641, -34837.1445312, 36524.5351562
3: -6634.8359375, 18059.1308594, -8194.8476562, 22387.1894531, -29022.0253906, 26253.9765625
4: -21026.6679688, 12626.4082031, -26064.4238281, 15510.4072266, -36537.0742188, 38690.8320312

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7975584, upper bound: 19342.8141399
time: 1.00 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8158441, upper bound: 19342.9181034
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -14864.4892578, 10995.2011719, -15843.4423828, 11758.1757812, -26622.6640625, 26838.6425781
1: -12031.5576172, 10648.3916016, -12824.6943359, 11386.0986328, -23417.6542969, 23473.0859375
2: -17551.9765625, 11632.7685547, -18705.6972656, 12435.0947266, -29987.0683594, 30338.4648438
3: -6072.9863281, 16529.3085938, -6487.5097656, 17652.9628906, -23725.9472656, 23016.8183594
4: -19286.0390625, 11497.1298828, -20550.6035156, 12283.4970703, -31569.5332031, 32047.7285156

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7688422, upper bound: 19342.8498503
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7655416, upper bound: 19342.7819118
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -14864.4892578, 10995.2011719, -18374.6835938, 13570.6152344, -28435.1035156, 29369.8847656
1: -12031.5576172, 10648.3916016, -14873.8886719, 13141.1816406, -25172.7382812, 25522.2792969
2: -17551.9765625, 11632.7685547, -21705.7109375, 14352.2373047, -31904.2128906, 33338.4765625
3: -6072.9863281, 16529.3085938, -7491.6044922, 20461.2734375, -26534.2578125, 24020.9140625
4: -19286.0390625, 11497.1298828, -23838.8925781, 14172.1142578, -33458.1523438, 35336.0234375

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7688422, upper bound: 19342.8498503
time: 1.17 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7655416, upper bound: 19342.7819118
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -16812.8476562, 12580.3085938, -15718.0712891, 11676.7431641, -28489.5898438, 28298.3789062
1: -13607.9619141, 12177.2802734, -12722.0048828, 11307.5380859, -24915.4960938, 24899.2851562
2: -19901.6875000, 13285.3291016, -18563.1523438, 12349.7265625, -32251.4140625, 31848.4804688
3: -6896.1435547, 18776.3144531, -6441.1611328, 17524.4394531, -24420.5820312, 25217.4765625
4: -21866.5410156, 13114.0878906, -20394.2265625, 12198.0615234, -34064.5976562, 33508.3046875

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -16812.8476562, 12580.3085938, -18250.3476562, 13489.9199219, -30302.7675781, 30830.6562500
1: -13607.9619141, 12177.2802734, -14772.1347656, 13062.9941406, -26670.9550781, 26949.4121094
2: -19901.6875000, 13285.3291016, -21564.0292969, 14267.4531250, -34169.1406250, 34849.3593750
3: -6896.1435547, 18776.3144531, -7445.7050781, 20333.3242188, -27229.4687500, 26222.0195312
4: -21866.5410156, 13114.0878906, -23683.5273438, 14087.1650391, -35953.7070312, 36797.6171875

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -14814.0175781, 10957.7373047, -17902.4296875, 13280.4335938, -28094.4511719, 28860.1640625
1: -11990.6953125, 10612.1904297, -14494.2519531, 12861.5820312, -24852.2753906, 25106.4414062
2: -17492.8339844, 11593.3730469, -21140.4648438, 14046.8632812, -31539.6972656, 32733.8378906
3: -6052.0712891, 16473.4824219, -7331.5644531, 19963.2187500, -26015.2890625, 23805.0468750
4: -19220.9550781, 11458.0722656, -23227.8398438, 13885.0107422, -33105.9648438, 34685.9062500

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7564889, upper bound: 19342.8304195
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7546088, upper bound: 19342.7767029
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -14814.0175781, 10957.7373047, -20387.7480469, 15051.3378906, -29865.3554688, 31345.4843750
1: -11990.6953125, 10612.1904297, -16506.4101562, 14576.4003906, -26567.0917969, 27118.5937500
2: -17492.8339844, 11593.3730469, -24085.1835938, 15921.8701172, -33414.6992188, 35678.5546875
3: -6052.0712891, 16473.4824219, -8314.4375000, 22714.5605469, -28766.6308594, 24787.9199219
4: -19220.9550781, 11458.0722656, -26455.3378906, 15732.7304688, -34953.6835938, 37913.4101562

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7564889, upper bound: 19342.8304195
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7546088, upper bound: 19342.7767029
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -16769.9335938, 12548.1259766, -17775.9589844, 13201.5117188, -29971.4453125, 30324.0839844
1: -13573.2539062, 12146.4414062, -14390.2578125, 12785.3564453, -26358.6093750, 26536.6992188
2: -19851.6152344, 13251.6005859, -20998.3085938, 13963.8261719, -33815.4414062, 34249.9062500
3: -6878.2021484, 18728.8554688, -7286.8564453, 19836.6269531, -26714.8281250, 26015.7109375
4: -21811.3222656, 13080.4687500, -23070.3398438, 13801.8339844, -35613.1562500, 36150.8085938

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7966980, upper bound: 19342.8120156
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8145666, upper bound: 19342.9150038
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -16769.9335938, 12548.1259766, -20262.7109375, 14972.5595703, -31742.4921875, 32810.8359375
1: -13573.2539062, 12146.4414062, -16403.8652344, 14500.2880859, -28073.5429688, 28550.3027344
2: -19851.6152344, 13251.6005859, -23944.9375000, 15838.9501953, -35690.5664062, 37196.5390625
3: -6878.2021484, 18728.8554688, -8269.8339844, 22588.7265625, -29466.9296875, 26998.6894531
4: -21811.3222656, 13080.4687500, -26299.7988281, 15649.8173828, -37461.1406250, 39380.2656250

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7966980, upper bound: 19342.8120156
time: 0.91 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.8145666, upper bound: 19342.9150038
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -16857.2656250, 12396.5947266, -15056.7158203, 11192.8984375, -28050.1640625, 27453.3066406
1: -13645.6601562, 12005.1806641, -12188.0019531, 10835.3554688, -24481.0117188, 24193.1835938
2: -19913.0097656, 13110.3173828, -17779.6972656, 11834.6269531, -31747.6367188, 30890.0156250
3: -6855.3081055, 18727.6835938, -6171.5786133, 16782.1386719, -23637.4433594, 24899.2617188
4: -21875.3027344, 12953.9511719, -19531.1875000, 11688.9472656, -33564.2460938, 32485.1367188

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7588602, upper bound: 19342.7763523
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417527
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -16857.2656250, 12396.5947266, -16761.7890625, 12390.8437500, -29248.1093750, 29158.3769531
1: -13645.6601562, 12005.1806641, -13570.1767578, 11999.9882812, -25645.6445312, 25575.3574219
2: -19913.0097656, 13110.3173828, -19770.4765625, 13100.1796875, -33013.1875000, 32880.7929688
3: -6855.3081055, 18727.6835938, -6831.2573242, 18646.0644531, -25501.3710938, 25558.9394531
4: -21875.3027344, 12953.9511719, -21706.3125000, 12932.9042969, -34808.1914062, 34660.2617188

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7588602, upper bound: 19342.7763523
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417527
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -18910.3105469, 13909.8837891, -14997.0361328, 11148.1376953, -30058.4472656, 28906.9160156
1: -15310.9765625, 13469.1982422, -12139.8349609, 10792.0888672, -26103.0664062, 25609.0312500
2: -22338.8164062, 14708.9052734, -17709.8496094, 11787.0322266, -34125.8437500, 32418.7539062
3: -7690.5825195, 21028.7187500, -6145.7866211, 16715.9316406, -24406.5117188, 27174.5058594
4: -24542.2343750, 14542.1298828, -19454.2636719, 11641.7519531, -36183.9843750, 33996.3945312

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.1068669
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.7942067
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -18910.3105469, 13909.8837891, -16692.5546875, 12338.8134766, -31249.1250000, 30602.4335938
1: -15310.9765625, 13469.1982422, -13514.1777344, 11949.7714844, -27260.7460938, 26983.3730469
2: -22338.8164062, 14708.9052734, -19689.1152344, 13044.9208984, -35383.7382812, 34398.0195312
3: -7690.5825195, 21028.7187500, -6801.4321289, 18569.0410156, -26259.6210938, 27830.1503906
4: -24542.2343750, 14542.1298828, -21616.7675781, 12878.1123047, -37420.3476562, 36158.8984375

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.1068669
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.7942067
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -17024.1152344, 12509.9833984, -15606.0537109, 11584.5068359, -28608.6210938, 28116.0351562
1: -13781.0664062, 12115.2919922, -12632.6718750, 11216.7431641, -24997.8085938, 24747.9648438
2: -20109.8281250, 13230.3906250, -18429.0195312, 12249.3554688, -32359.1816406, 31659.4101562
3: -6920.8989258, 18910.1015625, -6389.1093750, 17391.9609375, -24312.8593750, 25299.2089844
4: -22092.6113281, 13073.0644531, -20245.9375000, 12099.3750000, -34191.9843750, 33319.0000000

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7611823, upper bound: 19342.7888746
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417045
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -17024.1152344, 12509.9833984, -17297.7792969, 12774.3125000, -29798.4277344, 29807.7617188
1: -13781.0664062, 12115.2919922, -14003.4941406, 12373.9101562, -26154.9746094, 26118.7832031
2: -20109.8281250, 13230.3906250, -20402.9375000, 13506.7187500, -33616.5390625, 33633.3281250
3: -6920.8989258, 18910.1015625, -7044.4960938, 19240.1933594, -26161.0917969, 25954.5976562
4: -22092.6113281, 13073.0644531, -22402.8925781, 13335.5048828, -35428.1093750, 35475.9531250

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7611823, upper bound: 19342.7888746
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417045
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -19078.7167969, 14029.7988281, -15558.7011719, 11549.6123047, -30628.3281250, 29588.4980469
1: -15447.2617188, 13584.5312500, -12594.4218750, 11183.0644531, -26630.3222656, 26178.9492188
2: -22538.5058594, 14834.5244141, -18373.6015625, 12212.3828125, -34750.8906250, 33208.1250000
3: -7760.8652344, 21215.3808594, -6369.4028320, 17339.6992188, -25100.5644531, 27584.7832031
4: -24762.8320312, 14667.9560547, -20184.9179688, 12062.6484375, -36825.4765625, 34852.8710938

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.1115472
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.8131094
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -19078.7167969, 14029.7988281, -17245.4531250, 12735.6972656, -31814.4140625, 31275.2519531
1: -15447.2617188, 13584.5312500, -13961.1611328, 12336.6455078, -27783.9023438, 27545.6914062
2: -22538.5058594, 14834.5244141, -20341.5175781, 13465.7988281, -36004.3046875, 35176.0312500
3: -7760.8652344, 21215.3808594, -7022.6938477, 19182.3281250, -26943.1933594, 28238.0742188
4: -24762.8320312, 14667.9560547, -22335.3183594, 13294.8339844, -38057.6640625, 37003.2695312

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.1115472
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.8131094
time: 1.02 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.27 seconds
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7697455, upper bound: 19342.8510653
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7691589, upper bound: 19342.7928483
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7697455, upper bound: 19342.8510653
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7691589, upper bound: 19342.7928483
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8352184, upper bound: 19342.9246218
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7575661, upper bound: 19342.8325564
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7575726, upper bound: 19342.7883602
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7575661, upper bound: 19342.8325564
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7575726, upper bound: 19342.7883602
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7975584, upper bound: 19342.8141399
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8158441, upper bound: 19342.9181034
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7975584, upper bound: 19342.8141399
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8158441, upper bound: 19342.9181034
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7688422, upper bound: 19342.8498503
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7655416, upper bound: 19342.7819118
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7688422, upper bound: 19342.8498503
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7655416, upper bound: 19342.7819118
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8348799, upper bound: 19342.9219887
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7564889, upper bound: 19342.8304195
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7546088, upper bound: 19342.7767029
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7564889, upper bound: 19342.8304195
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7546088, upper bound: 19342.7767029
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7966980, upper bound: 19342.8120156
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8145666, upper bound: 19342.9150038
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7966980, upper bound: 19342.8120156
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.8145666, upper bound: 19342.9150038
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7588602, upper bound: 19342.7763523
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417527
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7588602, upper bound: 19342.7763523
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417527
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.1068669
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.7942067
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.1068669
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.5068728, upper bound: 19342.7942067
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7611823, upper bound: 19342.7888746
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417045
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7611823, upper bound: 19342.7888746
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19342.7726446, upper bound: 19342.8417045
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.1115472
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.8131094
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.1115472
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.27
Output dim: 3, lower bound: -19341.8224056, upper bound: 19342.8131094
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7752310
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7666871
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7697512, upper bound: 19342.7748939
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.27
Output dim: 3, lower bound: -19342.7658847, upper bound: 19342.7658848
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.1250000, mid=0.1250000, abs_max=24289.171875
rel_dist={3: [-19343.089144719655, 19343.089144719655]}

## Binary Search with IS_dual_ind Result
status: None
Maximum delta epsilon: None
execution time: 1100.20 seconds
