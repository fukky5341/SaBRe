## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 7905.840511004298


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1418.0635986, 6938.8701172, -1418.0635986, 6938.8701172, -8356.9335938, 8356.9335938)
1: (-2204.8723145, 8037.7197266, -2204.8723145, 8037.7197266, -10242.5898438, 10242.5898438)
2: (-1915.0596924, 8291.7822266, -1915.0596924, 8291.7822266, -10206.8398438, 10206.8398438)
3: (-2938.4826660, 6100.9482422, -2938.4826660, 6100.9482422, -9039.4306641, 9039.4296875)
4: (-2028.4676514, 6489.7880859, -2028.4676514, 6489.7880859, -8518.2558594, 8518.2558594)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.79 + 2.18 = 2.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -7905.9195702, upper bound: 7905.9195702

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8454562, upper bound: 7905.8516154
time: 0.77 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9195702, upper bound: 7905.9195702
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.58 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -7905.8454562, upper bound: 7905.8516154
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -7905.9195702, upper bound: 7905.9195702

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1420.3673096, 6959.3759766, -1415.9025879, 6927.8105469, -8348.1777344, 8375.2783203
1: -2208.0717773, 8062.2138672, -2201.5107422, 8024.9301758, -10233.0019531, 10263.7246094
2: -1918.3492432, 8314.4892578, -1912.1636963, 8278.6279297, -10196.9755859, 10226.6523438
3: -2944.0207520, 6121.2465820, -2934.0737305, 6091.4467773, -9035.4667969, 9055.3183594
4: -2030.6054688, 6507.3784180, -2025.4677734, 6479.6557617, -8510.2617188, 8532.8447266

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7982941, upper bound: 7905.7982941
time: 5.78 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.7982941, upper bound: 7905.8516154
time: 0.73 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1417.8948975, 6938.0708008, -1418.0635986, 6938.8701172, -8356.7646484, 8356.1328125
1: -2204.6132812, 8036.7939453, -2204.8723145, 8037.7197266, -10242.3300781, 10241.6650391
2: -1914.8319092, 8290.8251953, -1915.0596924, 8291.7822266, -10206.6132812, 10205.8828125
3: -2938.1418457, 6100.2441406, -2938.4826660, 6100.9482422, -9039.0888672, 9038.7265625
4: -2028.2309570, 6489.0375977, -2028.4676514, 6489.7880859, -8518.0185547, 8517.5048828

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.6528661, upper bound: 7905.6392523
time: 0.77 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9194300, upper bound: 7905.9194300
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.34 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.34
Output dim: 3, lower bound: -7905.7982941, upper bound: 7905.7982941
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 3, lower bound: -7905.7982941, upper bound: 7905.8516154
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.34
Output dim: 3, lower bound: -7905.6528661, upper bound: 7905.6392523
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.34
Output dim: 3, lower bound: -7905.9194300, upper bound: 7905.9194300

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1420.3673096, 6959.3759766, -1417.8948975, 6938.0708008, -8358.4365234, 8377.2705078
1: -2208.0717773, 8062.2138672, -2204.6132812, 8036.7939453, -10244.8652344, 10266.8271484
2: -1918.3492432, 8314.4892578, -1914.8319092, 8290.8251953, -10209.1718750, 10229.3212891
3: -2944.0207520, 6121.2465820, -2938.1418457, 6100.2441406, -9044.2646484, 9059.3847656
4: -2030.6054688, 6507.3784180, -2028.2309570, 6489.0375977, -8519.6425781, 8535.6083984

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.7949934, upper bound: 7905.8515724
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.7982941, upper bound: 7905.8515724
time: 0.92 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1417.8948975, 6938.0708008, -1417.9202881, 6938.1694336, -8356.0644531, 8355.9902344
1: -2204.6132812, 8036.7939453, -2204.6499023, 8036.9096680, -10241.5205078, 10241.4423828
2: -1914.8319092, 8290.8251953, -1914.8666992, 8290.9443359, -10205.7753906, 10205.6914062
3: -2938.1418457, 6100.2441406, -2938.1799316, 6100.3325195, -9038.4726562, 9038.4238281
4: -2028.2309570, 6489.0375977, -2028.2618408, 6489.1333008, -8517.3632812, 8517.2998047

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8484659, upper bound: 7905.8416873
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8484659, upper bound: 7905.9192245
time: 0.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.39 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -7905.7949934, upper bound: 7905.8515724
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -7905.7982941, upper bound: 7905.8515724
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -7905.8484659, upper bound: 7905.8416873
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.39
Output dim: 3, lower bound: -7905.8484659, upper bound: 7905.9192245

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1356.6103516, 6651.5166016, -1374.3956299, 6726.3129883, -8082.9233398, 8025.9121094
1: -2109.5114746, 7705.0864258, -2137.4582520, 7791.2509766, -9900.7607422, 9842.5429688
2: -1832.4570312, 7945.9467773, -1856.4031982, 8037.8144531, -9870.2714844, 9802.3496094
3: -2812.5036621, 5849.0297852, -2849.1826172, 5914.0810547, -8726.5820312, 8698.2119141
4: -1939.8824463, 6217.8129883, -1967.1600342, 6291.1254883, -8231.0078125, 8184.9726562

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8407939, upper bound: 7905.8492604
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8407939, upper bound: 7905.8515724
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1490.2896729, 7309.3105469, -1412.1768799, 6910.5566406, -8400.8466797, 8721.4873047
1: -2318.3837891, 8467.1357422, -2195.7385254, 8004.9702148, -10323.3535156, 10662.8740234
2: -2015.3281250, 8734.9775391, -1907.0101318, 8258.0576172, -10273.3857422, 10641.9873047
3: -3100.8979492, 6430.8603516, -2926.8828125, 6076.4589844, -9177.3544922, 9357.7431641
4: -2146.1738281, 6842.0102539, -2020.4287109, 6463.6396484, -8609.8134766, 8862.4394531

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8454036, upper bound: 7905.8492604
time: 1.19 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8454036, upper bound: 7905.8515724
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1417.8948975, 6938.0708008, -1420.1962891, 6958.5302734, -8376.4238281, 8358.2666016
1: -2204.6132812, 8036.7939453, -2207.8044434, 8061.2363281, -10265.8496094, 10244.5976562
2: -1914.8319092, 8290.8251953, -1918.1204834, 8313.4785156, -10228.3105469, 10208.9433594
3: -2938.1418457, 6100.2441406, -2943.6616211, 6120.5073242, -9058.6494141, 9043.9052734
4: -2028.2309570, 6489.0375977, -2030.3612061, 6506.5917969, -8534.8203125, 8519.3984375

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.6871563, upper bound: 7905.6669644
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8448522, upper bound: 7905.8393653
time: 2.60 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1417.8948975, 6938.0708008, -1417.7513428, 6937.3691406, -8355.2636719, 8355.8212891
1: -2204.6132812, 8036.7939453, -2204.3894043, 8035.9814453, -10240.5937500, 10241.1826172
2: -1914.8319092, 8290.8251953, -1914.6395264, 8289.9873047, -10204.8183594, 10205.4648438
3: -2938.1418457, 6100.2441406, -2937.8386230, 6099.6279297, -9037.7695312, 9038.0830078
4: -2028.2309570, 6489.0375977, -2028.0251465, 6488.3833008, -8516.6142578, 8517.0625000

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8460977, upper bound: 7905.9192074
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8484229, upper bound: 7905.9192074
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.23 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.8407939, upper bound: 7905.8492604
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.8407939, upper bound: 7905.8515724
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.8454036, upper bound: 7905.8492604
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.8454036, upper bound: 7905.8515724
NS_A2_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.6871563, upper bound: 7905.6669644
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.8448522, upper bound: 7905.8393653
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.8460977, upper bound: 7905.9192074
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.23
Output dim: 3, lower bound: -7905.8484229, upper bound: 7905.9192074

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1356.6103516, 6651.5166016, -1354.7572021, 6633.2016602, -7989.8120117, 8006.2739258
1: -2109.5114746, 7705.0864258, -2107.0371094, 7683.1611328, -9792.6689453, 9812.1210938
2: -1832.4570312, 7945.9467773, -1829.8061523, 7925.7944336, -9758.2519531, 9775.7529297
3: -2812.5036621, 5849.0297852, -2808.1650391, 5830.9775391, -8643.4775391, 8657.1923828
4: -1939.8824463, 6217.8129883, -1938.6386719, 6202.5288086, -8142.4111328, 8156.4501953

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8236456, upper bound: 7905.8434465
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8235453, upper bound: 7905.8324398
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1356.6103516, 6651.5166016, -1488.7181396, 7292.3354492, -8648.9453125, 8140.2343750
1: -2109.5114746, 7705.0864258, -2316.3369141, 8446.9912109, -10556.5000000, 10021.4238281
2: -1832.4570312, 7945.9467773, -2013.0596924, 8717.2656250, -10549.7226562, 9959.0068359
3: -2812.5036621, 5849.0297852, -3097.4409180, 6414.4174805, -9226.9189453, 8946.4697266
4: -1939.8824463, 6217.8129883, -2145.5039062, 6828.2817383, -8768.1640625, 8363.3164062

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8058860, upper bound: 7905.7973972
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8022047, upper bound: 7905.7948871
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1490.2896729, 7309.3105469, -1354.7572021, 6633.2016602, -8123.4912109, 8664.0673828
1: -2318.3837891, 8467.1357422, -2107.0371094, 7683.1611328, -10001.5439453, 10574.1718750
2: -2015.3281250, 8734.9775391, -1829.8061523, 7925.7944336, -9941.1230469, 10564.7832031
3: -3100.8979492, 6430.8603516, -2808.1650391, 5830.9775391, -8931.8730469, 9239.0244141
4: -2146.1738281, 6842.0102539, -1938.6386719, 6202.5288086, -8348.7021484, 8780.6484375

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7915949, upper bound: 7905.8167421
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7888288, upper bound: 7905.7948871
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1490.2896729, 7309.3105469, -1488.7181396, 7292.3354492, -8782.6250000, 8798.0283203
1: -2318.3837891, 8467.1357422, -2316.3369141, 8446.9912109, -10765.3750000, 10783.4726562
2: -2015.3281250, 8734.9775391, -2013.0596924, 8717.2656250, -10732.5937500, 10748.0371094
3: -3100.8979492, 6430.8603516, -3097.4409180, 6414.4174805, -9515.3144531, 9528.3007812
4: -2146.1738281, 6842.0102539, -2145.5039062, 6828.2817383, -8974.4531250, 8987.5136719

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8260634, upper bound: 7905.8434601
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8260379, upper bound: 7905.8325211
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1409.7531738, 6897.8256836, -1437.3120117, 7041.2612305, -8451.0146484, 8335.1376953
1: -2191.9685059, 7990.1586914, -2234.2416992, 8157.3022461, -10349.2685547, 10224.3974609
2: -1903.8172607, 8242.9042969, -1940.9697266, 8412.5517578, -10316.3691406, 10183.8740234
3: -2921.5124512, 6065.0097656, -2979.4514160, 6193.9687500, -9115.4814453, 9044.4609375
4: -2016.8961182, 6451.7158203, -2055.3950195, 6584.7563477, -8601.6494141, 8507.1103516

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B2_B1

### Relational analysis result of NS_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8448226, upper bound: 7905.8351252
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2

### Relational analysis result of NS_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8448226, upper bound: 7905.8393653
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1354.7572021, 6633.2016602, -1374.2531738, 6725.6157227, -8080.3730469, 8007.4550781
1: -2107.0371094, 7683.1611328, -2137.2363281, 7790.4443359, -9897.4785156, 9820.3964844
2: -1829.8061523, 7925.7944336, -1856.2115479, 8036.9814453, -9866.7851562, 9782.0058594
3: -2808.1650391, 5830.9775391, -2848.8820801, 5913.4687500, -8721.6318359, 8679.8583984
4: -1938.6386719, 6202.5288086, -1966.9558105, 6290.4746094, -8229.1113281, 8169.4843750

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9178480, upper bound: 7905.9176672
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9178480, upper bound: 7905.9192074
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1488.7181396, 7292.3354492, -1412.0338135, 6909.8564453, -8398.5742188, 8704.3691406
1: -2316.3369141, 8446.9912109, -2195.5148926, 8004.1591797, -10320.4960938, 10642.5058594
2: -2013.0596924, 8717.2656250, -1906.8176270, 8257.2167969, -10270.2763672, 10624.0830078
3: -3097.4409180, 6414.4174805, -2926.5795898, 6075.8442383, -9173.2832031, 9340.9970703
4: -2145.5039062, 6828.2817383, -2020.2229004, 6462.9843750, -8608.4873047, 8848.5029297

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9194281, upper bound: 7905.9176672
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9194281, upper bound: 7905.9192074
time: 0.93 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.43 seconds
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8236456, upper bound: 7905.8434465
NS_A1_B2_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8235453, upper bound: 7905.8324398
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8058860, upper bound: 7905.7973972
NS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8022047, upper bound: 7905.7948871
NS_A1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.7915949, upper bound: 7905.8167421
NS_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.7888288, upper bound: 7905.7948871
NS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8260634, upper bound: 7905.8434601
NS_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8260379, upper bound: 7905.8325211
NS_A2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8448226, upper bound: 7905.8351252
NS_A2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.8448226, upper bound: 7905.8393653
NS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.9178480, upper bound: 7905.9176672
NS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.9178480, upper bound: 7905.9192074
NS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.9194281, upper bound: 7905.9176672
NS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.43
Output dim: 3, lower bound: -7905.9194281, upper bound: 7905.9192074

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1344.5661621, 6591.9770508, -1336.8316650, 6544.3681641, -7888.9345703, 7928.8081055
1: -2090.7731934, 7636.1826172, -2079.1437988, 7580.3442383, -9671.1162109, 9715.3261719
2: -1816.2232666, 7874.9306641, -1805.6147461, 7819.8188477, -9636.0419922, 9680.5439453
3: -2787.8684082, 5797.1489258, -2771.3903809, 5753.5688477, -8541.4375000, 8568.5390625
4: -1922.8516846, 6162.4970703, -1913.2061768, 6120.0092773, -8042.8608398, 8075.7031250

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8233153, upper bound: 7905.8320215
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8233153, upper bound: 7905.8324398
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1477.9378662, 7248.1206055, -1469.9975586, 7199.3388672, -8677.2763672, 8718.1181641
1: -2299.1611328, 8396.3808594, -2287.2053223, 8339.5058594, -10638.6660156, 10683.5839844
2: -1998.6892090, 8662.0664062, -1987.7940674, 8606.5537109, -10605.2431641, 10649.8603516
3: -3075.7434082, 6377.8032227, -3059.3171387, 6333.8754883, -9409.6191406, 9437.1162109
4: -2128.8684082, 6785.3935547, -2119.2722168, 6742.3613281, -8871.2285156, 8904.6640625

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8266050, upper bound: 7905.8320609
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8266050, upper bound: 7905.8325211
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -1366.1077881, 6685.3476562, -1373.3664551, 6731.1484375, -8097.2563477, 8058.7138672
1: -2124.5935059, 7743.7895508, -2135.4536133, 7797.6064453, -9922.1992188, 9879.2431641
2: -1845.1940918, 7989.0595703, -1854.8958740, 8041.5722656, -9886.7636719, 9843.9550781
3: -2832.3251953, 5878.2324219, -2848.0207520, 5920.3554688, -8752.6806641, 8726.2529297
4: -1955.6772461, 6253.1909180, -1965.0045166, 6293.8999023, -8249.5771484, 8218.1953125

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8350323
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8351252
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -1404.0832520, 6870.5263672, -1508.0450439, 7395.2983398, -8799.3818359, 8378.5703125
1: -2183.1679688, 7958.5917969, -2345.8632812, 8567.0546875, -10750.2226562, 10304.4550781
2: -1896.0607910, 8210.3955078, -2039.0346680, 8838.2910156, -10734.3515625, 10249.4277344
3: -2910.3500977, 6041.4150391, -3138.1450195, 6507.3017578, -9417.6523438, 9179.5585938
4: -2009.1583252, 6426.5234375, -2172.1298828, 6923.4028320, -8932.5585938, 8598.6513672

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8391741
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8393653
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1354.7572021, 6633.2016602, -1354.6160889, 6632.5112305, -7987.2685547, 7987.8173828
1: -2107.0371094, 7683.1611328, -2106.8168945, 7682.3637695, -9789.3974609, 9789.9755859
2: -1829.8061523, 7925.7944336, -1829.6164551, 7924.9697266, -9754.7753906, 9755.4111328
3: -2808.1650391, 5830.9775391, -2807.8679199, 5830.3725586, -8638.5351562, 8638.8417969
4: -1938.6386719, 6202.5288086, -1938.4359131, 6201.8837891, -8140.5214844, 8140.9643555

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9071071, upper bound: 7905.9090257
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9064289, upper bound: 7905.9064846
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1354.7572021, 6633.2016602, -1488.5679932, 7291.5976562, -8646.3544922, 8121.7695312
1: -2107.0371094, 7683.1611328, -2316.1030273, 8446.1386719, -10553.1728516, 9999.2636719
2: -1829.8061523, 7925.7944336, -2012.8577881, 8716.3828125, -10546.1894531, 9938.6513672
3: -2808.1650391, 5830.9775391, -3097.1269531, 6413.7729492, -9221.9365234, 8928.1015625
4: -1938.6386719, 6202.5288086, -2145.2912598, 6827.5952148, -8766.2333984, 8347.8203125

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8772859, upper bound: 7905.8591964
time: 0.90 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1488.7181396, 7292.3354492, -1354.6160889, 6632.5112305, -8121.2290039, 8646.9511719
1: -2316.3369141, 8446.9912109, -2106.8168945, 7682.3637695, -9998.7001953, 10553.8076172
2: -2013.0596924, 8717.2656250, -1829.6164551, 7924.9697266, -9938.0292969, 10546.8818359
3: -3097.4409180, 6414.4174805, -2807.8679199, 5830.3725586, -8927.8125000, 9222.2832031
4: -2145.5039062, 6828.2817383, -1938.4359131, 6201.8837891, -8347.3876953, 8766.7158203

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8593498, upper bound: 7905.8755947
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8566547, upper bound: 7905.8563834
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1488.7181396, 7292.3354492, -1488.5679932, 7291.5976562, -8780.3154297, 8780.9033203
1: -2316.3369141, 8446.9912109, -2316.1030273, 8446.1386719, -10762.4755859, 10763.0937500
2: -2013.0596924, 8717.2656250, -2012.8577881, 8716.3828125, -10729.4423828, 10730.1220703
3: -3097.4409180, 6414.4174805, -3097.1269531, 6413.7729492, -9511.2138672, 9511.5419922
4: -2145.5039062, 6828.2817383, -2145.2912598, 6827.5952148, -8973.0986328, 8973.5722656

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9107168, upper bound: 7905.9121624
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9102277, upper bound: 7905.9103882
time: 0.75 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.75 seconds
NS_A1_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8233153, upper bound: 7905.8320215
NS_A1_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8233153, upper bound: 7905.8324398
NS_A1_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8266050, upper bound: 7905.8320609
NS_A1_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8266050, upper bound: 7905.8325211
NS_A2_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8350323
NS_A2_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8351252
NS_A2_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8391741
NS_A2_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8420131, upper bound: 7905.8393653
NS_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.9071071, upper bound: 7905.9090257
NS_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.9064289, upper bound: 7905.9064846
NS_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8772859, upper bound: 7905.8591964
NS_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
NS_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8593498, upper bound: 7905.8755947
NS_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.8566547, upper bound: 7905.8563834
NS_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.9107168, upper bound: 7905.9121624
NS_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.75
Output dim: 3, lower bound: -7905.9102277, upper bound: 7905.9103882

## BFS NS instance: NS_A2_B2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1345.7708740, 6588.2011719, -1373.3664551, 6731.1484375, -8076.9194336, 7961.5673828
1: -2093.0986328, 7630.9760742, -2135.4536133, 7797.6064453, -9890.7050781, 9766.4296875
2: -1817.6602783, 7872.3276367, -1854.8958740, 8041.5722656, -9859.2294922, 9727.2236328
3: -2789.8317871, 5791.5747070, -2848.0207520, 5920.3554688, -8710.1875000, 8639.5957031
4: -1926.2274170, 6160.9560547, -1965.0045166, 6293.8999023, -8220.1269531, 8125.9604492

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8358368, upper bound: 7905.8183233
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8243736, upper bound: 7905.8177021
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1480.0391846, 7249.1552734, -1373.3664551, 6731.1484375, -8211.1875000, 8622.5205078
1: -2302.8530273, 8396.9814453, -2135.4536133, 7797.6064453, -10100.4589844, 10532.4355469
2: -2001.3205566, 8665.9130859, -1854.8958740, 8041.5722656, -10042.8906250, 10520.8085938
3: -3079.7402344, 6376.7231445, -2848.0207520, 5920.3554688, -9000.0957031, 9224.7441406
4: -2133.4658203, 6788.3632812, -1965.0045166, 6293.8999023, -8427.3652344, 8753.3681641

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B1_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7806661, upper bound: 7905.7919442
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B1_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7773783, upper bound: 7905.7873945
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1345.7708740, 6588.2011719, -1508.0450439, 7395.2983398, -8741.0693359, 8096.2460938
1: -2093.0986328, 7630.9760742, -2345.8632812, 8567.0546875, -10660.1533203, 9976.8388672
2: -1817.6602783, 7872.3276367, -2039.0346680, 8838.2910156, -10655.9511719, 9911.3623047
3: -2789.8317871, 5791.5747070, -3138.1450195, 6507.3017578, -9297.1337891, 8929.7177734
4: -1926.2274170, 6160.9560547, -2172.1298828, 6923.4028320, -8849.6298828, 8333.0849609

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8044943, upper bound: 7905.7825640
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7773783, upper bound: 7905.7747664
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1480.0391846, 7249.1552734, -1508.0450439, 7395.2983398, -8875.3378906, 8757.1992188
1: -2302.8530273, 8396.9814453, -2345.8632812, 8567.0546875, -10869.9082031, 10742.8447266
2: -2001.3205566, 8665.9130859, -2039.0346680, 8838.2910156, -10839.6113281, 10704.9462891
3: -3079.7402344, 6376.7231445, -3138.1450195, 6507.3017578, -9587.0419922, 9514.8671875
4: -2133.4658203, 6788.3632812, -2172.1298828, 6923.4028320, -9056.8681641, 8960.4921875

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8358368, upper bound: 7905.8208868
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8243736, upper bound: 7905.8201079
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1351.9011230, 6618.5156250, -1331.6431885, 6514.5444336, -7866.4453125, 7950.1586914
1: -2102.6101074, 7666.1279297, -2071.2211914, 7545.5781250, -9648.1865234, 9737.3496094
2: -1825.9504395, 7908.3979492, -1798.6011963, 7785.2475586, -9611.1982422, 9706.9990234
3: -2802.2744141, 5818.1157227, -2760.4765625, 5726.9858398, -8529.2597656, 8578.5917969
4: -1934.6791992, 6189.0146484, -1906.5758057, 6093.2622070, -8027.9409180, 8095.5903320

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9030630, upper bound: 7905.9052880
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9045400, upper bound: 7905.9061266
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1345.7708740, 6588.2011719, -1365.8489990, 6682.8852539, -8028.6557617, 7954.0502930
1: -2093.0986328, 7630.9760742, -2124.1577148, 7740.9482422, -9834.0468750, 9755.1328125
2: -1817.6602783, 7872.3276367, -1844.5887451, 7986.3125000, -9803.9726562, 9716.9160156
3: -2789.8317871, 5791.5747070, -2832.2580566, 5876.8613281, -8666.6933594, 8623.8300781
4: -1926.2274170, 6160.9560547, -1956.2878418, 6251.6777344, -8177.9047852, 8117.2431641

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9064289, upper bound: 7905.9064846
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9064289, upper bound: 7905.9064846
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1319.1337891, 6453.7539062, -1467.8157959, 7187.7011719, -8506.8349609, 7921.5698242
1: -2051.5351562, 7475.6044922, -2283.7592773, 8325.8095703, -10377.3447266, 9759.3632812
2: -1781.8770752, 7711.8789062, -1984.9123535, 8592.2216797, -10374.0986328, 9696.7910156
3: -2734.5717773, 5674.7021484, -3053.8154297, 6322.4379883, -9057.0097656, 8728.5175781
4: -1887.9228516, 6036.2797852, -2115.3879395, 6730.6474609, -8618.5703125, 8151.6679688

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1611.2354736, 7917.2836914, -1440.6199951, 7064.0878906, -8675.3232422, 9357.9033203
1: -2506.7958984, 9172.3925781, -2242.0681152, 8182.5419922, -10689.3369141, 11414.4609375
2: -2176.4353027, 9460.9306641, -1948.4403076, 8443.7255859, -10620.1611328, 11409.3691406
3: -3348.1708984, 6958.2841797, -2998.8730469, 6211.5268555, -9559.6972656, 9957.1572266
4: -2311.5429688, 7402.7578125, -2076.7839355, 6612.8574219, -8924.3984375, 9479.5419922

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1467.9643555, 7188.4331055, -1318.9925537, 6453.0629883, -7921.0273438, 8507.4257812
1: -2283.9904785, 8326.6552734, -2051.3151855, 7474.8051758, -9758.7939453, 10377.9707031
2: -1985.1116943, 8593.0957031, -1781.6875000, 7711.0517578, -9696.1630859, 10374.7802734
3: -3054.1262207, 6323.0781250, -2734.2736816, 5674.0971680, -8728.2226562, 9057.3505859
4: -2115.5983887, 6731.3286133, -1887.7207031, 6035.6337891, -8151.2324219, 8619.0488281

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1440.7645264, 7064.8017578, -1611.0910645, 7916.5795898, -9357.3437500, 8675.8916016
1: -2242.2939453, 8183.3676758, -2506.5708008, 9171.5791016, -11413.8710938, 10689.9375000
2: -1948.6346436, 8444.5781250, -2176.2414551, 9460.0869141, -11408.7216797, 10620.8193359
3: -2999.1765137, 6212.1494141, -3347.8664551, 6957.6655273, -9956.8417969, 9560.0156250
4: -2076.9895020, 6613.5209961, -2311.3364258, 7402.0991211, -9479.0888672, 8924.8574219

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1485.8624268, 7277.6259766, -1465.5086670, 7173.0009766, -8658.8623047, 8743.1347656
1: -2311.9128418, 8429.9482422, -2280.3764648, 8308.7011719, -10620.6132812, 10710.3232422
2: -2009.2037354, 8699.8720703, -1981.7153320, 8576.1181641, -10585.3212891, 10681.5869141
3: -3091.5705566, 6401.5639648, -3049.7087402, 6310.0590820, -9401.6298828, 9451.2714844
4: -2141.5551758, 6814.7724609, -2113.3886719, 6718.5991211, -8860.1542969, 8928.1611328

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9071076, upper bound: 7905.9082945
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9084402, upper bound: 7905.9096226
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1480.0391846, 7249.1552734, -1502.5008545, 7356.3417969, -8836.3808594, 8751.6552734
1: -2302.8530273, 8396.9814453, -2337.5598145, 8521.4882812, -10824.3417969, 10734.5410156
2: -2001.3205566, 8665.9130859, -2031.3627930, 8794.6201172, -10795.9394531, 10697.2753906
3: -3079.7402344, 6376.7231445, -3126.8662109, 6472.6699219, -9552.4101562, 9503.5888672
4: -2133.4658203, 6788.3632812, -2166.4226074, 6890.2861328, -9023.7519531, 8954.7861328

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9106843, upper bound: 7905.9103882
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9106843, upper bound: 7905.9103882
time: 0.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.33 seconds
NS_A2_B2_B1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8358368, upper bound: 7905.8183233
NS_A2_B2_B1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8243736, upper bound: 7905.8177021
NS_A2_B2_B1_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.7806661, upper bound: 7905.7919442
NS_A2_B2_B1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.7773783, upper bound: 7905.7873945
NS_A2_B2_B1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8044943, upper bound: 7905.7825640
NS_A2_B2_B1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.7773783, upper bound: 7905.7747664
NS_A2_B2_B1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8358368, upper bound: 7905.8208868
NS_A2_B2_B1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8243736, upper bound: 7905.8201079
NS_A2_B2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9030630, upper bound: 7905.9052880
NS_A2_B2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9045400, upper bound: 7905.9061266
NS_A2_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9064289, upper bound: 7905.9064846
NS_A2_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9064289, upper bound: 7905.9064846
NS_A2_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
NS_A2_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
NS_A2_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
NS_A2_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8702387, upper bound: 7905.8568783
NS_A2_B2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
NS_A2_B2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
NS_A2_B2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
NS_A2_B2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.8570854, upper bound: 7905.8690557
NS_A2_B2_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9071076, upper bound: 7905.9082945
NS_A2_B2_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9084402, upper bound: 7905.9096226
NS_A2_B2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9106843, upper bound: 7905.9103882
NS_A2_B2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.33
Output dim: 3, lower bound: -7905.9106843, upper bound: 7905.9103882

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -1350.3795166, 6610.7358398, -1314.8684082, 6428.9794922, -7779.3588867, 7925.6040039
1: -2100.2497559, 7657.1030273, -2045.1997070, 7446.3413086, -9546.5908203, 9702.3027344
2: -1823.8956299, 7899.1499023, -1775.9396973, 7683.5561523, -9507.4521484, 9675.0898438
3: -2799.0810547, 5811.2636719, -2725.2854004, 5651.5786133, -8450.6591797, 8536.5488281
4: -1932.5242920, 6181.7963867, -1882.8164062, 6013.8276367, -7946.3505859, 8064.6113281

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9025660
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9052880
time: 1.08 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -1346.7021484, 6591.9560547, -1338.6271973, 6544.8603516, -7891.5625000, 7930.5830078
1: -2094.6108398, 7635.4907227, -2082.1169434, 7580.8911133, -9675.5009766, 9717.6074219
2: -1819.0850830, 7876.9912109, -1808.1312256, 7822.1425781, -9641.2265625, 9685.1220703
3: -2792.0380859, 5795.5629883, -2775.1281738, 5754.7636719, -8546.8017578, 8570.6894531
4: -1927.7486572, 6165.0034180, -1916.8045654, 6123.1240234, -8050.8725586, 8081.8076172

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9045400, upper bound: 7905.9061266
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9045400, upper bound: 7905.9061266
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1331.7834473, 6515.2275391, -1365.8489990, 6682.8852539, -8014.6684570, 7881.0766602
1: -2071.4389648, 7546.3686523, -2124.1577148, 7740.9482422, -9812.3867188, 9670.5253906
2: -1798.7894287, 7786.0659180, -1844.5887451, 7986.3125000, -9785.1015625, 9630.6542969
3: -2760.7722168, 5727.5874023, -2832.2580566, 5876.8613281, -8637.6337891, 8559.8457031
4: -1906.7764893, 6093.9013672, -1956.2878418, 6251.6777344, -8158.4541016, 8050.1879883

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9008980
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9039711, upper bound: 7905.9039291
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1365.9666748, 6683.4672852, -1365.8489990, 6682.8852539, -8048.8520508, 8049.3164062
1: -2124.3415527, 7741.6210938, -2124.1577148, 7740.9482422, -9865.2890625, 9865.7773438
2: -1844.7468262, 7987.0083008, -1844.5887451, 7986.3125000, -9831.0595703, 9831.5966797
3: -2832.5041504, 5877.3666992, -2832.2580566, 5876.8613281, -8709.3652344, 8709.6240234
4: -1956.4533691, 6252.2177734, -1956.2878418, 6251.6777344, -8208.1308594, 8208.5048828

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9008980
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9039711, upper bound: 7905.9039291
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1319.1337891, 6453.7539062, -1454.2932129, 7119.5229492, -8438.6562500, 7908.0468750
1: -2051.5351562, 7475.6044922, -2262.6491699, 8246.8916016, -10298.4267578, 9738.2539062
2: -1781.8770752, 7711.8789062, -1966.6983643, 8510.8212891, -10292.6982422, 9678.5771484
3: -2734.5717773, 5674.7021484, -3025.5363770, 6262.6206055, -8997.1923828, 8700.2382812
4: -1887.9228516, 6036.2797852, -2095.8742676, 6667.1611328, -8555.0839844, 8132.1542969

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8609444, upper bound: 7905.8443201
time: 0.89 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8503315, upper bound: 7905.8329701
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1319.1337891, 6453.7539062, -1643.8918457, 8099.9887695, -9419.1220703, 8097.6455078
1: -2051.5351562, 7475.6044922, -2558.3510742, 9382.6582031, -11434.1923828, 10033.9550781
2: -1781.8770752, 7711.8789062, -2219.8386230, 9679.7890625, -11461.6660156, 9931.7177734
3: -2734.5717773, 5674.7021484, -3415.9858398, 7110.4516602, -9845.0234375, 9090.6875000
4: -1887.9228516, 6036.2797852, -2357.0246582, 7568.7529297, -9456.6757812, 8393.3046875

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8609444, upper bound: 7905.8443201
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8503315, upper bound: 7905.8329701
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1611.2354736, 7917.2836914, -1454.2932129, 7119.5229492, -8730.7587891, 9371.5771484
1: -2506.7958984, 9172.3925781, -2262.6491699, 8246.8916016, -10753.6875000, 11435.0410156
2: -2176.4353027, 9460.9306641, -1966.6983643, 8510.8212891, -10687.2568359, 11427.6289062
3: -3348.1708984, 6958.2841797, -3025.5363770, 6262.6206055, -9610.7910156, 9983.8203125
4: -2311.5429688, 7402.7578125, -2095.8742676, 6667.1611328, -8978.7041016, 9498.6318359

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8443741, upper bound: 7905.8353685
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8381296, upper bound: 7905.8263642
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1611.2354736, 7917.2836914, -1643.8918457, 8099.9887695, -9711.2246094, 9561.1757812
1: -2506.7958984, 9172.3925781, -2558.3510742, 9382.6582031, -11889.4531250, 11730.7431641
2: -2176.4353027, 9460.9306641, -2219.8386230, 9679.7890625, -11856.2246094, 11680.7685547
3: -3348.1708984, 6958.2841797, -3415.9858398, 7110.4516602, -10458.6230469, 10374.2695312
4: -2311.5429688, 7402.7578125, -2357.0246582, 7568.7529297, -9880.2958984, 9759.7802734

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8443741, upper bound: 7905.8353685
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8381296, upper bound: 7905.8263642
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1454.4432373, 7120.2641602, -1318.9925537, 6453.0629883, -7907.5063477, 8439.2568359
1: -2262.8833008, 8247.7490234, -2051.3151855, 7474.8051758, -9737.6884766, 10299.0644531
2: -1966.9002686, 8511.7080078, -1781.6875000, 7711.0517578, -9677.9521484, 10293.3935547
3: -3025.8510742, 6263.2670898, -2734.2736816, 5674.0971680, -8699.9453125, 8997.5410156
4: -2096.0874023, 6667.8500977, -1887.7207031, 6035.6337891, -8131.7211914, 8555.5703125

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8562743, upper bound: 7905.8731845
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8568659, upper bound: 7905.8733056
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1644.0371094, 8100.7001953, -1318.9925537, 6453.0629883, -8097.1000977, 9419.6923828
1: -2558.5776367, 9383.4843750, -2051.3151855, 7474.8051758, -10033.3828125, 11434.7988281
2: -2220.0339355, 9680.6416016, -1781.6875000, 7711.0517578, -9931.0859375, 11462.3291016
3: -3416.2919922, 7111.0766602, -2734.2736816, 5674.0971680, -9090.3867188, 9845.3505859
4: -2357.2326660, 7569.4184570, -1887.7207031, 6035.6337891, -8392.8662109, 9457.1386719

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8562743, upper bound: 7905.8731845
time: 0.66 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8568659, upper bound: 7905.8733056
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1454.4432373, 7120.2641602, -1611.0910645, 7916.5795898, -9371.0224609, 8731.3554688
1: -2262.8833008, 8247.7490234, -2506.5708008, 9171.5791016, -11434.4609375, 10754.3203125
2: -1966.9002686, 8511.7080078, -2176.2414551, 9460.0869141, -11426.9873047, 10687.9492188
3: -3025.8510742, 6263.2670898, -3347.8664551, 6957.6655273, -9983.5166016, 9611.1337891
4: -2096.0874023, 6667.8500977, -2311.3364258, 7402.0991211, -9498.1865234, 8979.1855469

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8331733, upper bound: 7905.8464265
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8254973, upper bound: 7905.8381270
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1644.0371094, 8100.7001953, -1611.0910645, 7916.5795898, -9560.6162109, 9711.7910156
1: -2558.5776367, 9383.4843750, -2506.5708008, 9171.5791016, -11730.1552734, 11890.0546875
2: -2220.0339355, 9680.6416016, -2176.2414551, 9460.0869141, -11680.1210938, 11856.8828125
3: -3416.2919922, 7111.0766602, -3347.8664551, 6957.6655273, -10373.9570312, 10458.9433594
4: -2357.2326660, 7569.4184570, -2311.3364258, 7402.0991211, -9759.3310547, 9880.7548828

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8544981, upper bound: 7905.8669068
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8546722, upper bound: 7905.8668932
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -1484.3714600, 7269.9804688, -1448.9838867, 7088.4560547, -8572.8261719, 8718.9628906
1: -2309.5998535, 8421.0830078, -2254.7456055, 8210.7070312, -10520.3066406, 10675.8251953
2: -2007.1877441, 8690.8017578, -1959.3790283, 8475.8173828, -10483.0029297, 10650.1806641
3: -3088.4580078, 6394.8540039, -3015.2421875, 6235.8686523, -9324.3261719, 9410.0957031
4: -2139.4545898, 6807.7045898, -2090.1215820, 6640.4321289, -8779.8867188, 8897.8251953

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9082945
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1480.5621338, 7250.8437500, -1474.8061523, 7215.9350586, -8696.4970703, 8725.6484375
1: -2303.7072754, 8399.0751953, -2294.7985840, 8358.6699219, -10662.3769531, 10693.8740234
2: -2002.1699219, 8668.0761719, -1994.2857666, 8627.8652344, -10630.0332031, 10662.3613281
3: -3081.0249023, 6378.7309570, -3069.1271973, 6348.9091797, -9429.9335938, 9447.8554688
4: -2134.3818359, 6790.3837891, -2126.7553711, 6759.8911133, -8894.2714844, 8917.1386719

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9096226
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1465.6551514, 7173.7197266, -1502.5008545, 7356.3417969, -8821.9970703, 8676.2197266
1: -2280.6037598, 8309.5312500, -2337.5598145, 8521.4882812, -10802.0917969, 10647.0908203
2: -1981.9117432, 8576.9775391, -2031.3627930, 8794.6201172, -10776.5302734, 10608.3398438
3: -3050.0144043, 6310.6870117, -3126.8662109, 6472.6699219, -9522.6845703, 9437.5527344
4: -2113.5964355, 6719.2680664, -2166.4226074, 6890.2861328, -9003.8818359, 8885.6904297

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9038308, upper bound: 7905.9036680
time: 0.90 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9082222, upper bound: 7905.9079532
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1502.6250000, 7356.9575195, -1502.5008545, 7356.3417969, -8858.9667969, 8859.4580078
1: -2337.7536621, 8522.2001953, -2337.5598145, 8521.4882812, -10859.2421875, 10859.7597656
2: -2031.5290527, 8795.3554688, -2031.3627930, 8794.6201172, -10826.1474609, 10826.7187500
3: -3127.1237793, 6473.2031250, -3126.8662109, 6472.6699219, -9599.7939453, 9600.0673828
4: -2166.5961914, 6890.8559570, -2166.4226074, 6890.2861328, -9056.8808594, 9057.2783203

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9038308, upper bound: 7905.9036680
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9082222, upper bound: 7905.9079532
time: 0.81 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 2.50 seconds
NS_A2_B2_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9025660
NS_A2_B2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9052880
NS_A2_B2_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9045400, upper bound: 7905.9061266
NS_A2_B2_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9045400, upper bound: 7905.9061266
NS_A2_B2_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9008980
NS_A2_B2_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9039711, upper bound: 7905.9039291
NS_A2_B2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9008980
NS_A2_B2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9039711, upper bound: 7905.9039291
NS_A2_B2_B2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8609444, upper bound: 7905.8443201
NS_A2_B2_B2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8503315, upper bound: 7905.8329701
NS_A2_B2_B2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8609444, upper bound: 7905.8443201
NS_A2_B2_B2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8503315, upper bound: 7905.8329701
NS_A2_B2_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8443741, upper bound: 7905.8353685
NS_A2_B2_B2_A1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8381296, upper bound: 7905.8263642
NS_A2_B2_B2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8443741, upper bound: 7905.8353685
NS_A2_B2_B2_A1_B2_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8381296, upper bound: 7905.8263642
NS_A2_B2_B2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8562743, upper bound: 7905.8731845
NS_A2_B2_B2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8568659, upper bound: 7905.8733056
NS_A2_B2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8562743, upper bound: 7905.8731845
NS_A2_B2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8568659, upper bound: 7905.8733056
NS_A2_B2_B2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8331733, upper bound: 7905.8464265
NS_A2_B2_B2_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8254973, upper bound: 7905.8381270
NS_A2_B2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8544981, upper bound: 7905.8669068
NS_A2_B2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.8546722, upper bound: 7905.8668932
NS_A2_B2_B2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
NS_A2_B2_B2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9082945
NS_A2_B2_B2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
NS_A2_B2_B2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9096226
NS_A2_B2_B2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9038308, upper bound: 7905.9036680
NS_A2_B2_B2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9082222, upper bound: 7905.9079532
NS_A2_B2_B2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9038308, upper bound: 7905.9036680
NS_A2_B2_B2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 2.50
Output dim: 3, lower bound: -7905.9082222, upper bound: 7905.9079532

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1335.1420898, 6532.8818359, -1314.8684082, 6428.9794922, -7764.1215820, 7847.7500000
1: -2076.6142578, 7566.7763672, -2045.1997070, 7446.3413086, -9522.9550781, 9611.9765625
2: -1803.3189697, 7806.6152344, -1775.9396973, 7683.5561523, -9486.8730469, 9582.5546875
3: -2767.0700684, 5742.6489258, -2725.2854004, 5651.5786133, -8418.6484375, 8467.9345703
4: -1910.9256592, 6109.5200195, -1882.8164062, 6013.8276367, -7924.7524414, 7992.3349609

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8723191, upper bound: 7905.8610525
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453412, upper bound: 7905.8504414
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1358.4777832, 6646.7275391, -1314.8684082, 6428.9794922, -7787.4570312, 7961.5957031
1: -2112.8789062, 7698.9931641, -2045.1997070, 7446.3413086, -9559.2207031, 9744.1904297
2: -1834.9521484, 7942.7983398, -1775.9396973, 7683.5561523, -9518.5078125, 9718.7382812
3: -2816.1284180, 5844.1425781, -2725.2854004, 5651.5786133, -8467.7070312, 8569.4277344
4: -1944.3863525, 6216.9951172, -1882.8164062, 6013.8276367, -7958.2128906, 8099.8095703

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8996353, upper bound: 7905.8919527
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8864966, upper bound: 7905.8908363
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1326.5993652, 6488.7387695, -1338.6271973, 6544.8603516, -7871.4599609, 7827.3662109
1: -2063.4631348, 7515.8281250, -2082.1169434, 7580.8911133, -9644.3544922, 9597.9443359
2: -1791.9418945, 7754.7373047, -1808.1312256, 7822.1425781, -9614.0839844, 9562.8681641
3: -2750.5661621, 5705.0986328, -2775.1281738, 5754.7636719, -8505.3300781, 8480.2246094
4: -1899.8647461, 6069.9565430, -1916.8045654, 6123.1240234, -8022.9877930, 7986.7607422

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9025660
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9061266
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1359.1451416, 6649.2783203, -1338.6271973, 6544.8603516, -7904.0053711, 7987.9052734
1: -2113.8220215, 7702.2299805, -2082.1169434, 7580.8911133, -9694.7128906, 9784.3466797
2: -1835.7200928, 7946.5229492, -1808.1312256, 7822.1425781, -9657.8623047, 9754.6542969
3: -2819.2255859, 5848.4287109, -2775.1281738, 5754.7636719, -8573.9892578, 8623.5556641
4: -1947.4655762, 6221.3120117, -1916.8045654, 6123.1240234, -8070.5883789, 8138.1162109

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9025660
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9061266
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1330.2600098, 6507.4536133, -1349.9918213, 6601.5253906, -7931.7851562, 7857.4453125
1: -2069.0764160, 7537.3505859, -2099.5520020, 7646.5717773, -9715.6484375, 9636.9023438
2: -1796.7321777, 7776.8266602, -1823.1550293, 7889.6572266, -9686.3896484, 9599.9814453
3: -2757.5778809, 5720.7363281, -2798.8859863, 5805.1484375, -8562.7255859, 8519.6220703
4: -1904.6204834, 6086.6860352, -1933.7924805, 6176.1953125, -8080.8154297, 8020.4780273

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8996055, upper bound: 7905.9017647
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8996055, upper bound: 7905.9017647
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1326.5993652, 6488.7387695, -1370.6485596, 6699.5668945, -8026.1660156, 7859.3872070
1: -2063.4631348, 7515.8281250, -2131.4555664, 7760.8862305, -9824.3496094, 9647.2822266
2: -1791.9418945, 7754.7373047, -1851.2833252, 8007.4863281, -9799.4267578, 9606.0185547
3: -2750.5661621, 5705.0986328, -2843.1682129, 5894.9799805, -8645.5458984, 8548.2666016
4: -1899.8647461, 6069.9565430, -1964.3258057, 6270.8295898, -8170.6933594, 8034.2822266

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9089798, upper bound: 7905.9048075
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9089798, upper bound: 7905.9048075
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1364.5198975, 6676.0488281, -1349.9918213, 6601.5253906, -7966.0454102, 8026.0405273
1: -2122.0971680, 7733.0141602, -2099.5520020, 7646.5717773, -9768.6689453, 9832.5664062
2: -1842.7913818, 7978.1962891, -1823.1550293, 7889.6572266, -9732.4482422, 9801.3515625
3: -2829.4604492, 5870.8256836, -2798.8859863, 5805.1484375, -8634.6093750, 8669.7109375
4: -1954.4017334, 6245.3334961, -1933.7924805, 6176.1953125, -8130.5966797, 8179.1254883

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9005711
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9008980
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1359.1451416, 6649.2783203, -1370.6485596, 6699.5668945, -8058.7119141, 8019.9257812
1: -2113.8220215, 7702.2299805, -2131.4555664, 7760.8862305, -9874.7080078, 9833.6855469
2: -1835.7200928, 7946.5229492, -1851.2833252, 8007.4863281, -9843.2060547, 9797.8066406
3: -2819.2255859, 5848.4287109, -2843.1682129, 5894.9799805, -8714.2050781, 8691.5966797
4: -1947.4655762, 6221.3120117, -1964.3258057, 6270.8295898, -8218.2939453, 8185.6376953

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9017917, upper bound: 7905.9007315
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9017917, upper bound: 7905.9039291
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1316.2799072, 6439.0737305, -1431.2357178, 7000.9624023, -8317.2412109, 7870.3095703
1: -2047.1110840, 7458.5844727, -2226.9191895, 8109.5117188, -10156.6230469, 9685.5029297
2: -1778.0233154, 7694.4902344, -1935.5538330, 8370.5742188, -10148.5976562, 9630.0439453
3: -2728.6870117, 5661.8461914, -2978.0781250, 6158.8706055, -8887.5556641, 8639.9238281
4: -1883.9667969, 6022.7724609, -2063.9436035, 6558.1611328, -8442.1269531, 8086.7158203

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8907617, upper bound: 7905.9086621
time: 0.69 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1310.4971924, 6410.4150391, -1468.5247803, 7186.2929688, -8496.7900391, 7878.9399414
1: -2038.1357422, 7425.3662109, -2284.5722656, 8324.5722656, -10362.7060547, 9709.9384766
2: -1770.2053223, 7660.4047852, -1985.6132812, 8591.3525391, -10361.5576172, 9646.0175781
3: -2716.9565430, 5636.7978516, -3055.6794434, 6322.9941406, -9039.9511719, 8692.4765625
4: -1876.0158691, 5996.2788086, -2117.2282715, 6731.4111328, -8607.4248047, 8113.5063477

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9055313, upper bound: 7905.9100687
time: 0.88 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9062473, upper bound: 7905.9096967
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1316.2799072, 6439.0737305, -1621.2150879, 7982.7719727, -9299.0507812, 8060.2890625
1: -2047.1110840, 7458.5844727, -2523.2197266, 9246.7050781, -11293.8164062, 9981.8037109
2: -1778.0233154, 7694.4902344, -2189.2004395, 9541.0351562, -11319.0585938, 9883.6904297
3: -2728.6870117, 5661.8461914, -3369.2475586, 7007.9106445, -9736.5947266, 9031.0937500
4: -1883.9667969, 6022.7724609, -2325.6552734, 7461.0625000, -9345.0292969, 8348.4277344

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8496093, upper bound: 7905.8399763
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8598061, upper bound: 7905.8442185
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1310.4971924, 6410.4150391, -1659.2530518, 8171.6835938, -9482.1806641, 8069.6679688
1: -2038.1357422, 7425.3662109, -2582.0695801, 9465.8945312, -11504.0283203, 10007.4355469
2: -1770.2053223, 7660.4047852, -2240.3735352, 9765.8349609, -11536.0400391, 9900.7783203
3: -2716.9565430, 5636.7978516, -3448.3723145, 7175.1396484, -9892.0957031, 9085.1679688
4: -1876.0158691, 5996.2788086, -2379.8181152, 7637.4550781, -9513.4707031, 8376.0966797

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8388914, upper bound: 7905.8296802
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8492876, upper bound: 7905.8327884
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1608.4379883, 7902.8422852, -1431.2357178, 7000.9624023, -8609.3974609, 9334.0781250
1: -2502.4677734, 9155.6435547, -2226.9191895, 8109.5117188, -10611.9794922, 11382.5615234
2: -2172.6577148, 9443.8339844, -1935.5538330, 8370.5742188, -10543.2324219, 11379.3876953
3: -3342.4155273, 6945.6528320, -2978.0781250, 6158.8706055, -9501.2861328, 9923.7304688
4: -2307.6804199, 7389.4941406, -2063.9436035, 6558.1611328, -8865.8417969, 9453.4355469

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8276262, upper bound: 7905.8572729
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8476173, upper bound: 7905.8702402
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1608.4379883, 7902.8422852, -1621.2150879, 7982.7719727, -9591.2080078, 9524.0566406
1: -2502.4677734, 9155.6435547, -2523.2197266, 9246.7050781, -11749.1718750, 11678.8623047
2: -2172.6577148, 9443.8339844, -2189.2004395, 9541.0351562, -11713.6914062, 11633.0341797
3: -3342.4155273, 6945.6528320, -3369.2475586, 7007.9106445, -10350.3242188, 10314.9003906
4: -2307.6804199, 7389.4941406, -2325.6552734, 7461.0625000, -9768.7431641, 9715.1445312

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8233642, upper bound: 7905.8180114
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8436872, upper bound: 7905.8347058
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1452.9550781, 7112.6347656, -1302.2220459, 6367.3945312, -7820.3491211, 8414.8564453
1: -2260.5737305, 8238.9033203, -2025.2980957, 7375.4243164, -9635.9980469, 10264.2011719
2: -1964.8876953, 8502.6552734, -1759.0294189, 7609.2138672, -9574.1005859, 10261.6845703
3: -3022.7426758, 6256.5708008, -2699.0654297, 5598.5864258, -8621.3291016, 8955.6337891
4: -2093.9907227, 6660.7958984, -1863.9566650, 5956.1147461, -8050.1054688, 8524.7509766

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9070869, upper bound: 7905.9059293
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9038094, upper bound: 7905.9009007
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1449.1027832, 7093.3740234, -1325.4963379, 6481.3603516, -7930.4628906, 8418.8701172
1: -2254.6386719, 8216.7568359, -2061.4982910, 7507.7749023, -9762.4130859, 10278.2548828
2: -1959.8360596, 8479.7919922, -1790.5513916, 7745.5366211, -9705.3730469, 10270.3417969
3: -3015.2775879, 6240.3730469, -2748.1235352, 5700.2153320, -8715.4921875, 8988.4951172
4: -2088.9035645, 6643.3823242, -1897.3507080, 6063.7163086, -8152.6196289, 8540.7333984

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9132654, upper bound: 7905.9098332
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9132654, upper bound: 7905.9098332
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1642.5543213, 8093.0312500, -1302.2220459, 6367.3945312, -8009.9487305, 9395.2529297
1: -2556.2749023, 9374.5888672, -2025.2980957, 7375.4243164, -9931.6962891, 11399.8867188
2: -2218.0283203, 9671.5458984, -1759.0294189, 7609.2138672, -9827.2421875, 11430.5742188
3: -3413.1872559, 7104.3579102, -2699.0654297, 5598.5864258, -9011.7734375, 9803.4238281
4: -2355.1450195, 7562.3422852, -1863.9566650, 5956.1147461, -8311.2578125, 9426.2978516

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8529875, upper bound: 7905.8645073
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1638.7366943, 8073.5068359, -1325.4963379, 6481.3603516, -8120.0971680, 9399.0029297
1: -2550.3623047, 9352.1357422, -2061.4982910, 7507.7749023, -10058.1337891, 11413.6328125
2: -2213.0195312, 9648.4248047, -1790.5513916, 7745.5366211, -9958.5566406, 11438.9765625
3: -3405.7844238, 7088.0400391, -2748.1235352, 5700.2153320, -9106.0000000, 9836.1621094
4: -2350.0678711, 7544.7875977, -1897.3507080, 6063.7163086, -8413.7832031, 9442.1386719

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8533631, upper bound: 7905.8645076
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8565430, upper bound: 7905.8720209
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1451.5858154, 7105.5522461, -1588.5780029, 7800.4985352, -9252.0830078, 8694.1298828
1: -2258.4548340, 8230.7031250, -2471.7424316, 9036.9472656, -11295.4013672, 10702.4453125
2: -1963.0410156, 8494.3076172, -2145.8383789, 9322.6748047, -11285.7158203, 10640.1435547
3: -3019.9716797, 6250.4023438, -3301.5434570, 6856.0825195, -9876.0537109, 9551.9453125
4: -2092.1330566, 6654.3334961, -2280.2414551, 7295.4428711, -9387.5742188, 8934.5722656

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8687713, upper bound: 7905.8541876
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8798378, upper bound: 7905.8607735
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1642.5543213, 8093.0312500, -1594.5260010, 7831.4750977, -9474.0273438, 9687.5576172
1: -2556.2749023, 9374.5888672, -2480.8796387, 9072.8701172, -11629.1435547, 11855.4677734
2: -2218.0283203, 9671.5458984, -2153.8598633, 9359.0683594, -11577.0966797, 11825.4052734
3: -3413.1872559, 7104.3579102, -3313.1601562, 6882.8906250, -10296.0781250, 10417.5175781
4: -2355.1450195, 7562.3422852, -2287.9597168, 7323.3574219, -9678.5000000, 9850.3017578

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8544981, upper bound: 7905.8667786
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8544981, upper bound: 7905.8668932
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1638.7366943, 8073.5068359, -1617.2574463, 7942.2695312, -9581.0058594, 9690.7646484
1: -2550.3623047, 9352.1357422, -2516.1965332, 9201.5732422, -11751.9326172, 11868.3291016
2: -2213.0195312, 9648.4248047, -2184.7646484, 9491.6357422, -11704.6552734, 11833.1894531
3: -3405.7844238, 7088.0400391, -3361.2006836, 6981.9604492, -10387.7451172, 10449.2402344
4: -2350.0678711, 7544.7875977, -2320.7592773, 7428.2260742, -9778.2929688, 9865.5468750

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8546722, upper bound: 7905.8667786
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8546722, upper bound: 7905.8668932
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1469.3736572, 7193.0908203, -1448.9838867, 7088.4560547, -8557.8281250, 8642.0742188
1: -2286.3291016, 8331.9570312, -2254.7456055, 8210.7070312, -10497.0361328, 10586.7011719
2: -1986.9096680, 8599.5888672, -1959.3790283, 8475.8173828, -10462.7246094, 10558.9677734
3: -3057.1472168, 6327.3847656, -3015.2421875, 6235.8686523, -9293.0146484, 9342.6259766
4: -2118.3220215, 6736.6372070, -2090.1215820, 6640.4321289, -8758.7529297, 8826.7578125

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8376759, upper bound: 7905.8224489
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8015266, upper bound: 7905.8109307
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1494.4822998, 7317.0791016, -1448.9838867, 7088.4560547, -8582.9365234, 8766.0625000
1: -2325.2832031, 8475.9072266, -2254.7456055, 8210.7070312, -10535.9902344, 10730.6513672
2: -2020.8891602, 8747.5039062, -1959.3790283, 8475.8173828, -10496.7050781, 10706.8828125
3: -3109.6459961, 6437.4658203, -3015.2421875, 6235.8686523, -9345.5136719, 9452.7080078
4: -2154.0253906, 6852.9208984, -2090.1215820, 6640.4321289, -8794.4570312, 8943.0410156

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9027051, upper bound: 7905.8950991
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8923360, upper bound: 7905.8937804
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1469.3736572, 7193.0908203, -1474.8061523, 7215.9350586, -8685.3076172, 8667.8964844
1: -2286.3291016, 8331.9570312, -2294.7985840, 8358.6699219, -10644.9980469, 10626.7548828
2: -1986.9096680, 8599.5888672, -1994.2857666, 8627.8652344, -10614.7744141, 10593.8750000
3: -3057.1472168, 6327.3847656, -3069.1271973, 6348.9091797, -9406.0546875, 9396.5087891
4: -2118.3220215, 6736.6372070, -2126.7553711, 6759.8911133, -8878.2119141, 8863.3925781

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1494.4822998, 7317.0791016, -1474.8061523, 7215.9350586, -8710.4169922, 8791.8837891
1: -2325.2832031, 8475.9072266, -2294.7985840, 8358.6699219, -10683.9531250, 10770.7060547
2: -2020.8891602, 8747.5039062, -1994.2857666, 8627.8652344, -10648.7539062, 10741.7900391
3: -3109.6459961, 6437.4658203, -3069.1271973, 6348.9091797, -9458.5537109, 9506.5927734
4: -2154.0253906, 6852.9208984, -2126.7553711, 6759.8911133, -8913.9160156, 8979.6757812

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8372383, upper bound: 7905.8445095
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8014170, upper bound: 7905.8342584
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1464.1616211, 7166.0761719, -1486.7606201, 7275.5112305, -8739.6728516, 8652.8339844
1: -2278.2878418, 8300.6708984, -2313.1350098, 8427.7792969, -10706.0673828, 10613.8056641
2: -1979.8935547, 8567.9091797, -2010.0672607, 8698.7373047, -10678.6289062, 10577.9765625
3: -3046.8996582, 6303.9790039, -3093.9216309, 6401.6611328, -9448.5595703, 9397.8964844
4: -2111.4936523, 6712.2026367, -2144.1887207, 6815.5258789, -8927.0175781, 8856.3906250

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9050230, upper bound: 7905.9042444
time: 0.99 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9050230, upper bound: 7905.9042444
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1460.3812256, 7146.9970703, -1507.2565918, 7372.9223633, -8833.3037109, 8654.2539062
1: -2272.4316406, 8278.7343750, -2344.7475586, 8541.3300781, -10813.7597656, 10623.4814453
2: -1974.9082031, 8545.2529297, -2037.9860840, 8815.7333984, -10790.6416016, 10583.2392578
3: -3039.5148926, 6287.9282227, -3137.9196777, 6491.0332031, -9530.5458984, 9425.8447266
4: -2106.4545898, 6694.9536133, -2174.5458984, 6909.5263672, -9015.9804688, 8869.4970703

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9122648, upper bound: 7905.9083084
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9122648, upper bound: 7905.9083084
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1501.1966553, 7349.6250000, -1486.7606201, 7275.5112305, -8776.7080078, 8836.3847656
1: -2335.5375977, 8513.6992188, -2313.1350098, 8427.7792969, -10763.3164062, 10826.8339844
2: -2029.5971680, 8786.6591797, -2010.0672607, 8698.7373047, -10728.3310547, 10796.7265625
3: -3124.1330566, 6466.7602539, -3093.9216309, 6401.6611328, -9525.7939453, 9560.6777344
4: -2164.5766602, 6884.0703125, -2144.1887207, 6815.5258789, -8980.1025391, 9028.2587891

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9038155, upper bound: 7905.9033891
time: 0.87 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9038155, upper bound: 7905.9036680
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1495.1397705, 7319.4711914, -1507.2565918, 7372.9223633, -8868.0625000, 8826.7275391
1: -2326.1962891, 8478.9824219, -2344.7475586, 8541.3300781, -10867.5244141, 10823.7285156
2: -2021.6225586, 8750.9150391, -2037.9860840, 8815.7333984, -10837.3554688, 10788.9013672
3: -3112.4714355, 6441.3955078, -3137.9196777, 6491.0332031, -9603.5039062, 9579.3125000
4: -2156.6884766, 6856.8862305, -2174.5458984, 6909.5263672, -9066.2148438, 9031.4306641

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9048468, upper bound: 7905.9037247
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9048468, upper bound: 7905.9079532
time: 0.73 seconds

## Summary of splitting at layer (split count: 8)
- Time for NS candidates: 2.50 seconds
NS_A2_B2_B2_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8723191, upper bound: 7905.8610525
NS_A2_B2_B2_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8453412, upper bound: 7905.8504414
NS_A2_B2_B2_A1_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8996353, upper bound: 7905.8919527
NS_A2_B2_B2_A1_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8864966, upper bound: 7905.8908363
NS_A2_B2_B2_A1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9025660
NS_A2_B2_B2_A1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9061266
NS_A2_B2_B2_A1_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9025660
NS_A2_B2_B2_A1_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9022702, upper bound: 7905.9061266
NS_A2_B2_B2_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8996055, upper bound: 7905.9017647
NS_A2_B2_B2_A1_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8996055, upper bound: 7905.9017647
NS_A2_B2_B2_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9089798, upper bound: 7905.9048075
NS_A2_B2_B2_A1_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9089798, upper bound: 7905.9048075
NS_A2_B2_B2_A1_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9005711
NS_A2_B2_B2_A1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8992918, upper bound: 7905.9008980
NS_A2_B2_B2_A1_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9017917, upper bound: 7905.9007315
NS_A2_B2_B2_A1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9017917, upper bound: 7905.9039291
NS_A2_B2_B2_A1_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8907617, upper bound: 7905.9086621
NS_A2_B2_B2_A1_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
NS_A2_B2_B2_A1_B2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9055313, upper bound: 7905.9100687
NS_A2_B2_B2_A1_B2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9062473, upper bound: 7905.9096967
NS_A2_B2_B2_A1_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8496093, upper bound: 7905.8399763
NS_A2_B2_B2_A1_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8598061, upper bound: 7905.8442185
NS_A2_B2_B2_A1_B2_A1_B2_B2_B1, status: Status.VERIFIED, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8388914, upper bound: 7905.8296802
NS_A2_B2_B2_A1_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8492876, upper bound: 7905.8327884
NS_A2_B2_B2_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8276262, upper bound: 7905.8572729
NS_A2_B2_B2_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8476173, upper bound: 7905.8702402
NS_A2_B2_B2_A1_B2_A2_B2_B1_B1, status: Status.VERIFIED, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8233642, upper bound: 7905.8180114
NS_A2_B2_B2_A1_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8436872, upper bound: 7905.8347058
NS_A2_B2_B2_A2_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9070869, upper bound: 7905.9059293
NS_A2_B2_B2_A2_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9038094, upper bound: 7905.9009007
NS_A2_B2_B2_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9132654, upper bound: 7905.9098332
NS_A2_B2_B2_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9132654, upper bound: 7905.9098332
NS_A2_B2_B2_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8529875, upper bound: 7905.8645073
NS_A2_B2_B2_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
NS_A2_B2_B2_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8533631, upper bound: 7905.8645076
NS_A2_B2_B2_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8565430, upper bound: 7905.8720209
NS_A2_B2_B2_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8687713, upper bound: 7905.8541876
NS_A2_B2_B2_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8798378, upper bound: 7905.8607735
NS_A2_B2_B2_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8544981, upper bound: 7905.8667786
NS_A2_B2_B2_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8544981, upper bound: 7905.8668932
NS_A2_B2_B2_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8546722, upper bound: 7905.8667786
NS_A2_B2_B2_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8546722, upper bound: 7905.8668932
NS_A2_B2_B2_A2_B2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8376759, upper bound: 7905.8224489
NS_A2_B2_B2_A2_B2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8015266, upper bound: 7905.8109307
NS_A2_B2_B2_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9027051, upper bound: 7905.8950991
NS_A2_B2_B2_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8923360, upper bound: 7905.8937804
NS_A2_B2_B2_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
NS_A2_B2_B2_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9053616, upper bound: 7905.9055000
NS_A2_B2_B2_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8372383, upper bound: 7905.8445095
NS_A2_B2_B2_A2_B2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.8014170, upper bound: 7905.8342584
NS_A2_B2_B2_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9050230, upper bound: 7905.9042444
NS_A2_B2_B2_A2_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9050230, upper bound: 7905.9042444
NS_A2_B2_B2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9122648, upper bound: 7905.9083084
NS_A2_B2_B2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9122648, upper bound: 7905.9083084
NS_A2_B2_B2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9038155, upper bound: 7905.9033891
NS_A2_B2_B2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9038155, upper bound: 7905.9036680
NS_A2_B2_B2_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9048468, upper bound: 7905.9037247
NS_A2_B2_B2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 2.50
Output dim: 3, lower bound: -7905.9048468, upper bound: 7905.9079532

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1299.5012207, 6353.3828125, -1293.6567383, 6322.0849609, -7621.5859375, 7647.0395508
1: -2021.0819092, 7359.1831055, -2012.1292725, 7322.7177734, -9343.7988281, 9371.3115234
2: -1755.3535156, 7592.6337891, -1747.3592529, 7556.0947266, -9311.4472656, 9339.9931641
3: -2693.4650879, 5586.3168945, -2681.3959961, 5558.4785156, -8251.9414062, 8267.7128906
4: -1860.1925049, 5943.2314453, -1852.5363770, 5914.7646484, -7774.9560547, 7795.7666016

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
time: 1.03 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1591.8702393, 7817.7373047, -1274.2856445, 6236.1928711, -7828.0629883, 9092.0214844
1: -2476.7712402, 9056.9365234, -1982.4417725, 7222.7407227, -9699.5097656, 11039.3779297
2: -2150.2719727, 9342.8144531, -1721.3051758, 7451.8305664, -9602.1015625, 11064.1191406
3: -3307.7036133, 6870.8745117, -2640.5988770, 5478.6894531, -8786.3916016, 9511.4736328
4: -2284.3007812, 7310.7485352, -1823.6091309, 5830.2744141, -8114.5751953, 9134.3564453

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1340.1763916, 6555.7343750, -1302.7363281, 6368.8486328, -7709.0244141, 7858.4702148
1: -2084.3928223, 7593.6889648, -2026.3232422, 7376.7661133, -9461.1591797, 9620.0117188
2: -1810.2260742, 7834.2006836, -1759.5667725, 7611.8154297, -9422.0410156, 9593.7675781
3: -2778.3894043, 5764.7529297, -2700.3879395, 5599.1762695, -8377.5654297, 8465.1396484
4: -1918.2866211, 6132.3623047, -1865.6062012, 5957.9536133, -7876.2397461, 7997.9672852

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
time: 0.67 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1369.0230713, 6697.4321289, -1295.3948975, 6331.5126953, -7700.5356445, 7992.8271484
1: -2129.7778320, 7758.4409180, -2015.1524658, 7333.6054688, -9463.3828125, 9773.5917969
2: -1849.3094482, 8004.6962891, -1749.8266602, 7567.2797852, -9416.5888672, 9754.5234375
3: -2841.9899902, 5892.2075195, -2685.1064453, 5566.4350586, -8408.4238281, 8577.3144531
4: -1962.1009521, 6268.3403320, -1855.0020752, 5923.2114258, -7885.3125000, 8123.3422852

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
time: 1.04 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -1315.0078125, 6429.6606445, -1338.6271973, 6544.8603516, -7859.8676758, 7768.2875977
1: -2045.4172363, 7447.1289062, -2082.1169434, 7580.8911133, -9626.3076172, 9529.2460938
2: -1776.1273193, 7684.3706055, -1808.1312256, 7822.1425781, -9598.2685547, 9492.5019531
3: -2725.5800781, 5652.1772461, -2775.1281738, 5754.7636719, -8480.3437500, 8427.3046875
4: -1883.0167236, 6014.4638672, -1916.8045654, 6123.1240234, -8006.1396484, 7931.2685547

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8916187, upper bound: 7905.8771485
time: 0.95 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8768044
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1338.7845459, 6545.6381836, -1338.6271973, 6544.8603516, -7883.6450195, 7884.2656250
1: -2082.3608398, 7581.7895508, -2082.1169434, 7580.8911133, -9663.2500000, 9663.9062500
2: -1808.3408203, 7823.0717773, -1808.1312256, 7822.1425781, -9630.4833984, 9631.2021484
3: -2775.4543457, 5755.4370117, -2775.1281738, 5754.7636719, -8530.2177734, 8530.5634766
4: -1917.0252686, 6123.8437500, -1916.8045654, 6123.1240234, -8040.1484375, 8040.6484375

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8916187, upper bound: 7905.8772666
time: 0.96 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8769095
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1350.1080322, 6602.1015625, -1338.6271973, 6544.8603516, -7894.9682617, 7940.7285156
1: -2099.7331543, 7647.2358398, -2082.1169434, 7580.8911133, -9680.6230469, 9729.3525391
2: -1823.3110352, 7890.3447266, -1808.1312256, 7822.1425781, -9645.4521484, 9698.4755859
3: -2799.1276855, 5805.6464844, -2775.1281738, 5754.7636719, -8553.8916016, 8580.7734375
4: -1933.9549561, 6176.7275391, -1916.8045654, 6123.1240234, -8057.0786133, 8093.5322266

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8717704, upper bound: 7905.8577448
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453412, upper bound: 7905.8475581
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1370.7913818, 6700.2807617, -1338.6271973, 6544.8603516, -7915.6518555, 8038.9082031
1: -2131.6772461, 7761.7099609, -2082.1169434, 7580.8911133, -9712.5683594, 9843.8271484
2: -1851.4731445, 8008.3378906, -1808.1312256, 7822.1425781, -9673.6123047, 9816.4687500
3: -2843.4611816, 5895.5913086, -2775.1281738, 5754.7636719, -8598.2246094, 8670.7197266
4: -1964.5223389, 6271.4833984, -1916.8045654, 6123.1240234, -8087.6464844, 8188.2880859

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8717704, upper bound: 7905.8658765
time: 0.70 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453412, upper bound: 7905.8571317
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1315.0078125, 6429.6606445, -1349.9918213, 6601.5253906, -7916.5327148, 7779.6523438
1: -2045.4172363, 7447.1289062, -2099.5520020, 7646.5717773, -9691.9892578, 9546.6806641
2: -1776.1273193, 7684.3706055, -1823.1550293, 7889.6572266, -9665.7841797, 9507.5253906
3: -2725.5800781, 5652.1772461, -2798.8859863, 5805.1484375, -8530.7285156, 8451.0625000
4: -1883.0167236, 6014.4638672, -1933.7924805, 6176.1953125, -8059.2109375, 7948.2563477

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8668871, upper bound: 7905.8477187
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8405464, upper bound: 7905.8463065
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1338.7845459, 6545.6381836, -1349.9918213, 6601.5253906, -7940.3100586, 7895.6298828
1: -2082.3608398, 7581.7895508, -2099.5520020, 7646.5717773, -9728.9326172, 9681.3408203
2: -1808.3408203, 7823.0717773, -1823.1550293, 7889.6572266, -9697.9980469, 9646.2255859
3: -2775.4543457, 5755.4370117, -2798.8859863, 5805.1484375, -8580.6025391, 8554.3222656
4: -1917.0252686, 6123.8437500, -1933.7924805, 6176.1953125, -8093.2197266, 8057.6357422

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8967401, upper bound: 7905.8858897
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8856234
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1315.0078125, 6429.6606445, -1370.6485596, 6699.5668945, -8014.5742188, 7800.3085938
1: -2045.4172363, 7447.1289062, -2131.4555664, 7760.8862305, -9806.3027344, 9578.5839844
2: -1776.1273193, 7684.3706055, -1851.2833252, 8007.4863281, -9783.6123047, 9535.6523438
3: -2725.5800781, 5652.1772461, -2843.1682129, 5894.9799805, -8620.5595703, 8495.3457031
4: -1883.0167236, 6014.4638672, -1964.3258057, 6270.8295898, -8153.8452148, 7978.7895508

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8668871, upper bound: 7905.8568827
time: 0.88 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8405464, upper bound: 7905.8568825
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1338.7845459, 6545.6381836, -1370.6485596, 6699.5668945, -8038.3515625, 7916.2861328
1: -2082.3608398, 7581.7895508, -2131.4555664, 7760.8862305, -9843.2470703, 9713.2451172
2: -1808.3408203, 7823.0717773, -1851.2833252, 8007.4863281, -9815.8261719, 9674.3525391
3: -2775.4543457, 5755.4370117, -2843.1682129, 5894.9799805, -8670.4335938, 8598.6054688
4: -1917.0252686, 6123.8437500, -1964.3258057, 6270.8295898, -8187.8544922, 8088.1694336

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8967400, upper bound: 7905.8887678
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8886638
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1350.1080322, 6602.1015625, -1349.9918213, 6601.5253906, -7951.6328125, 7952.0932617
1: -2099.7331543, 7647.2358398, -2099.5520020, 7646.5717773, -9746.3046875, 9746.7880859
2: -1823.3110352, 7890.3447266, -1823.1550293, 7889.6572266, -9712.9677734, 9713.4990234
3: -2799.1276855, 5805.6464844, -2798.8859863, 5805.1484375, -8604.2763672, 8604.5322266
4: -1933.9549561, 6176.7275391, -1933.7924805, 6176.1953125, -8110.1499023, 8110.5195312

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8634242, upper bound: 7905.8459574
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8359456, upper bound: 7905.8412120
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1370.7913818, 6700.2807617, -1349.9918213, 6601.5253906, -7972.3164062, 8050.2724609
1: -2131.6772461, 7761.7099609, -2099.5520020, 7646.5717773, -9778.2490234, 9861.2617188
2: -1851.4731445, 8008.3378906, -1823.1550293, 7889.6572266, -9741.1289062, 9831.4921875
3: -2843.4611816, 5895.5913086, -2798.8859863, 5805.1484375, -8648.6093750, 8694.4775391
4: -1964.5223389, 6271.4833984, -1933.7924805, 6176.1953125, -8140.7177734, 8205.2744141

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8964908, upper bound: 7905.8853247
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8834538, upper bound: 7905.8849334
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1350.1080322, 6602.1015625, -1370.6485596, 6699.5668945, -8049.6748047, 7972.7495117
1: -2099.7331543, 7647.2358398, -2131.4555664, 7760.8862305, -9860.6181641, 9778.6914062
2: -1823.3110352, 7890.3447266, -1851.2833252, 8007.4863281, -9830.7949219, 9741.6269531
3: -2799.1276855, 5805.6464844, -2843.1682129, 5894.9799805, -8694.1074219, 8648.8144531
4: -1933.9549561, 6176.7275391, -1964.3258057, 6270.8295898, -8204.7832031, 8141.0532227

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8634242, upper bound: 7905.8505421
time: 0.68 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8359456, upper bound: 7905.8445019
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1370.7913818, 6700.2807617, -1370.6485596, 6699.5668945, -8070.3579102, 8070.9291992
1: -2131.6772461, 7761.7099609, -2131.4555664, 7760.8862305, -9892.5634766, 9893.1660156
2: -1851.4731445, 8008.3378906, -1851.2833252, 8007.4863281, -9858.9560547, 9859.6191406
3: -2843.4611816, 5895.5913086, -2843.1682129, 5894.9799805, -8738.4414062, 8738.7597656
4: -1964.5223389, 6271.4833984, -1964.3258057, 6270.8295898, -8235.3505859, 8235.8085938

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8634242, upper bound: 7905.8546179
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8359456, upper bound: 7905.8507376
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -1304.8958740, 6382.5336914, -1414.5440674, 6918.2250977, -8223.1210938, 7797.0776367
1: -2029.4207764, 7393.1831055, -2200.9775391, 8013.9404297, -10043.3613281, 9594.1601562
2: -1762.6838379, 7627.1293945, -1913.0965576, 8272.1611328, -10034.8447266, 9540.2255859
3: -2705.4597168, 5612.7011719, -2944.2758789, 6087.2915039, -8792.7509766, 8556.9755859
4: -1867.9143066, 5970.4111328, -2040.7386475, 6481.8500977, -8349.7646484, 8011.1499023

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -1296.2481689, 6338.9018555, -1439.5019531, 7044.8168945, -8341.0644531, 7778.4033203
1: -2016.1689453, 7342.6479492, -2240.2971191, 8160.6870117, -10176.8554688, 9582.9453125
2: -1751.1224365, 7574.8872070, -1946.6583252, 8423.7880859, -10174.9101562, 9521.5449219
3: -2687.2097168, 5574.1464844, -2997.8378906, 6197.9345703, -8885.1435547, 8571.9824219
4: -1855.2282715, 5929.4526367, -2077.1193848, 6600.4819336, -8455.7099609, 8006.5717773

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 9

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -1282.5167236, 6271.1806641, -1358.0515137, 6654.6269531, -7937.1435547, 7629.2319336
1: -1994.4970703, 7264.6293945, -2111.8540039, 7707.8154297, -9702.3115234, 9376.4833984
2: -1732.6270752, 7494.4936523, -1835.9561768, 7953.8842773, -9686.5097656, 9330.4501953
3: -2660.0463867, 5516.6547852, -2833.1745605, 5859.8725586, -8519.9189453, 8349.8291016
4: -1836.9813232, 5867.7221680, -1966.1290283, 6235.2275391, -8072.2084961, 7833.8510742

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.8597299
time: 0.96 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.9096967
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -1304.1892090, 6378.0458984, -1454.6917725, 7116.0141602, -8420.2031250, 7832.7377930
1: -2028.3065186, 7387.9926758, -2263.2197266, 8243.6318359, -10271.9384766, 9651.2089844
2: -1761.7283936, 7621.9282227, -1967.3267822, 8508.4863281, -10270.2148438, 9589.2529297
3: -2704.0173340, 5609.1079102, -3028.9948730, 6263.7670898, -8967.7841797, 8638.1025391
4: -1867.1545410, 5966.6625977, -2099.2404785, 6668.4252930, -8535.5800781, 8065.9028320

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.8597299
time: 0.94 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.9096967
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -1287.9564209, 6298.2470703, -1474.3027344, 7265.2460938, -8553.2021484, 7772.5498047
1: -2002.9377441, 7295.9833984, -2292.6687012, 8414.8525391, -10417.7900391, 9588.6523438
2: -1739.9718018, 7526.6699219, -1991.5612793, 8681.3466797, -10421.3183594, 9518.2314453
3: -2671.0854492, 5540.2695312, -3068.9299316, 6384.5439453, -9055.6289062, 8609.1962891
4: -1844.4549561, 5892.7011719, -2125.3950195, 6793.4667969, -8637.9189453, 8018.0961914

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8496093, upper bound: 7905.8398724
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8495997, upper bound: 7905.8399763
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1310.0152588, 6406.9277344, -1604.0250244, 7894.5678711, -9204.5810547, 8010.9521484
1: -2037.3486328, 7421.4643555, -2496.5974121, 9145.2763672, -11182.6230469, 9918.0615234
2: -1769.6016846, 7656.2690430, -2166.5007324, 9437.1787109, -11206.7802734, 9822.7695312
3: -2715.8295898, 5634.3417969, -3336.1088867, 6934.0786133, -9649.9082031, 8970.4501953
4: -1875.1636963, 5993.3510742, -2303.2355957, 7382.3476562, -9257.5117188, 8296.5869141

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8384038, upper bound: 7905.8291458
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8583933, upper bound: 7905.8434199
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -1304.1892090, 6378.0458984, -1642.9173584, 8087.9111328, -9392.1005859, 8020.9633789
1: -2028.3065186, 7387.9926758, -2556.7651367, 9369.5351562, -11397.8398438, 9944.7548828
2: -1761.7283936, 7621.9282227, -2218.7717285, 9667.1142578, -11428.8427734, 9840.7001953
3: -2704.0173340, 5609.1079102, -3416.7124023, 7104.8183594, -9808.8359375, 9025.8203125
4: -1867.1545410, 5966.6625977, -2358.3518066, 7562.4912109, -9429.6455078, 8325.0146484

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8489204, upper bound: 7905.8326640
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8492876, upper bound: 7905.8327884
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -1606.9351807, 7895.1181641, -1414.7432861, 6916.6416016, -8523.5761719, 9309.8613281
1: -2500.1367188, 9146.6855469, -2201.3303223, 8011.7641602, -10511.9003906, 11348.0156250
2: -2170.6284180, 9434.6679688, -1913.2580566, 8270.5156250, -10441.1435547, 11347.9257812
3: -3339.2695312, 6938.8701172, -2943.6440430, 6084.8339844, -9424.1025391, 9882.5136719
4: -2305.5615234, 7382.3505859, -2040.7150879, 6480.1547852, -8785.7158203, 9423.0625000

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8276262, upper bound: 7905.8555007
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8276262, upper bound: 7905.8572729
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -1603.2388916, 7875.8032227, -1440.9696045, 7046.6123047, -8649.8515625, 9316.7724609
1: -2494.4438477, 9124.4667969, -2242.0490723, 8162.6157227, -10657.0585938, 11366.5146484
2: -2165.7985840, 9411.8994141, -1948.7175293, 8425.5068359, -10591.3056641, 11360.6171875
3: -3332.1862793, 6922.8178711, -2998.4528809, 6200.0688477, -9532.2548828, 9921.2705078
4: -2300.7749023, 7365.1826172, -2077.9895020, 6601.8842773, -8902.6572266, 9443.1699219

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8476173, upper bound: 7905.8702402
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8476173, upper bound: 7905.8702402
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -1603.2388916, 7875.8032227, -1629.6333008, 8020.1835938, -9623.4228516, 9505.4365234
1: -2494.4438477, 9124.4667969, -2536.2866211, 9290.3798828, -11784.8222656, 11660.7529297
2: -2165.7985840, 9411.8994141, -2200.7141113, 9586.4199219, -11752.2187500, 11612.6132812
3: -3332.1862793, 6922.8178711, -3387.4428711, 7042.8227539, -10375.0087891, 10310.2597656
4: -2300.7749023, 7365.1826172, -2338.2626953, 7498.0698242, -9798.8447266, 9703.4453125

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8234755, upper bound: 7905.8120729
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_A1_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B2_A2_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8234755, upper bound: 7905.8347058
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1450.0969238, 7097.9218750, -1279.2634277, 6249.6245117, -7699.7216797, 8377.1835938
1: -2256.1447754, 8221.8574219, -1989.7108154, 7238.9174805, -9495.0615234, 10211.5683594
2: -1961.0279541, 8485.2529297, -1728.0113525, 7469.7480469, -9430.7763672, 10213.2646484
3: -3016.8613281, 6243.7036133, -2651.7514648, 5495.3662109, -8512.2275391, 8895.4550781
4: -2090.0356445, 6647.2783203, -1832.1188965, 5847.6782227, -7937.7133789, 8479.3945312

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9056257, upper bound: 7905.9045529
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9056257, upper bound: 7905.9059293
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1444.5924072, 7071.0195312, -1314.7360840, 6424.5932617, -7869.1855469, 8385.7558594
1: -2247.5781250, 8190.7153320, -2044.6175537, 7441.9624023, -9689.5410156, 10235.3310547
2: -1953.5792236, 8453.1669922, -1775.7174072, 7678.6259766, -9632.2050781, 10228.8837891
3: -3005.6711426, 6220.2421875, -2725.9064941, 5650.7563477, -8656.4277344, 8946.1474609
4: -2082.3793945, 6622.3193359, -1883.4223633, 6011.9501953, -8094.3295898, 8505.7402344

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9037655, upper bound: 7905.9004054
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9037655, upper bound: 7905.9009007
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1438.0020752, 7036.0058594, -1325.4963379, 6481.3603516, -7919.3623047, 8361.5019531
1: -2237.3669434, 8150.0585938, -2061.4982910, 7507.7749023, -9745.1416016, 10211.5566406
2: -1944.6695557, 8411.7373047, -1790.5513916, 7745.5366211, -9690.2060547, 10202.2880859
3: -2991.5034180, 6189.2866211, -2748.1235352, 5700.2153320, -8691.7187500, 8937.4101562
4: -2072.9194336, 6589.9409180, -1897.3507080, 6063.7163086, -8136.6357422, 8487.2919922

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9094449, upper bound: 7905.8937816
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8987859, upper bound: 7905.8941269
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1463.3156738, 7161.4741211, -1325.4963379, 6481.3603516, -7944.6757812, 8486.9707031
1: -2276.6735840, 8295.7480469, -2061.4982910, 7507.7749023, -9784.4482422, 10357.2460938
2: -1978.9250488, 8561.3896484, -1790.5513916, 7745.5366211, -9724.4619141, 10351.9394531
3: -3044.5202637, 6300.6923828, -2748.1235352, 5700.2153320, -8744.7353516, 9048.8154297
4: -2108.9802246, 6707.5708008, -1897.3507080, 6063.7163086, -8172.6962891, 8604.9218750

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9051740, upper bound: 7905.9062771
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9037655, upper bound: 7905.9039687
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1495.6046143, 7374.7290039, -1273.8703613, 6226.3295898, -7721.9340820, 8648.5996094
1: -2325.6206055, 8541.8642578, -1981.0780029, 7212.5419922, -9538.1621094, 10522.9423828
2: -2020.3125000, 8811.0087891, -1720.9329834, 7441.1240234, -9461.4345703, 10531.9404297
3: -3112.8715820, 6480.5927734, -2641.4025879, 5476.8256836, -8589.6953125, 9121.9941406
4: -2154.9738770, 6894.2866211, -1824.3984375, 5825.8496094, -7980.8232422, 8718.6855469

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 28

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7997300, upper bound: 7905.8268131
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7924470, upper bound: 7905.8151094
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1625.3269043, 8004.6455078, -1295.9708252, 6335.2822266, -7960.6093750, 9300.6162109
1: -2529.5947266, 9272.9511719, -2015.5538330, 7338.3422852, -9867.9365234, 11288.5048828
2: -2195.2778320, 9567.4726562, -1750.6267090, 7571.0375977, -9766.3154297, 11318.0996094
3: -3379.9743652, 7030.3842773, -2686.2187500, 5571.1113281, -8951.0849609, 9716.6025391
4: -2332.6782227, 7483.4599609, -1855.1596680, 5926.7163086, -8259.3945312, 9338.6171875

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1491.0224609, 7351.1381836, -1296.8406982, 6338.5380859, -7829.5600586, 8647.9785156
1: -2318.5317383, 8514.6611328, -2016.8148193, 7342.9584961, -9661.4902344, 10531.4755859
2: -2014.2500000, 8783.0654297, -1752.0467529, 7575.4140625, -9589.6640625, 10535.1103516
3: -3103.8259277, 6460.6401367, -2689.9960938, 5577.2436523, -8681.0693359, 9150.6367188
4: -2148.8239746, 6872.9697266, -1857.4808350, 5932.0688477, -8080.8925781, 8730.4501953

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8529875, upper bound: 7905.8645076
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8529875, upper bound: 7905.8645073
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1621.5325928, 7985.3100586, -1319.2145996, 6448.9243164, -8070.4565430, 9304.5244141
1: -2523.7287598, 9250.7109375, -2051.7065430, 7470.3383789, -9994.0673828, 11302.4160156
2: -2190.3024902, 9544.5732422, -1782.1134033, 7707.0122070, -9897.3144531, 11326.6865234
3: -3372.6127930, 7014.1894531, -2735.2285156, 5672.5371094, -9045.1484375, 9749.4150391
4: -2327.6230469, 7466.0449219, -1888.5266113, 6034.0986328, -8361.7216797, 9354.5712891

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8720209
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B1_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1450.0969238, 7097.9218750, -1572.0181885, 7715.5107422, -9165.6074219, 8669.9394531
1: -2256.1447754, 8221.8574219, -2446.0532227, 8938.3847656, -11194.5292969, 10667.9101562
2: -1961.0279541, 8485.2529297, -2123.4790039, 9221.7919922, -11182.8203125, 10608.7324219
3: -3016.8613281, 6243.7036133, -3266.8515625, 6781.4169922, -9798.2783203, 9510.5546875
4: -2090.0356445, 6647.2783203, -2256.8728027, 7216.7954102, -9306.8291016, 8904.1484375

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8431974
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8541876
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1446.2467041, 7078.6665039, -1595.0705566, 7827.8715820, -9274.1181641, 8673.7373047
1: -2250.2114258, 8199.7187500, -2481.8674316, 9068.8740234, -11319.0859375, 10681.5859375
2: -1955.9772949, 8462.3906250, -2154.7907715, 9356.2099609, -11312.1855469, 10617.1816406
3: -3009.4001465, 6227.5107422, -3315.4851074, 6881.7587891, -9891.1582031, 9542.9960938
4: -2084.9494629, 6629.8686523, -2290.0666504, 7323.0615234, -9408.0107422, 8919.9355469

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8431974
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8607735
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1627.6329346, 8015.9062500, -1594.5260010, 7831.4750977, -9459.1074219, 9610.4326172
1: -2533.1064453, 9285.1699219, -2480.8796387, 9072.8701172, -11605.9755859, 11766.0478516
2: -2197.8559570, 9580.0595703, -2153.8598633, 9359.0683594, -11556.9238281, 11733.9199219
3: -3381.9667969, 7036.8022461, -3313.1601562, 6882.8906250, -10264.8574219, 10349.9619141
4: -2334.1520996, 7491.1982422, -2287.9597168, 7323.3574219, -9657.5097656, 9779.1582031

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8027376, upper bound: 7905.8268185
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8177655
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1651.7623291, 8134.5854492, -1594.5260010, 7831.4750977, -9483.2373047, 9729.1113281
1: -2570.5659180, 9423.0849609, -2480.8796387, 9072.8701172, -11643.4355469, 11903.9628906
2: -2230.6286621, 9721.8144531, -2153.8598633, 9359.0683594, -11589.6972656, 11875.6738281
3: -3433.0656738, 7142.9433594, -3313.1601562, 6882.8906250, -10315.9560547, 10456.1025391
4: -2368.8823242, 7603.1733398, -2287.9597168, 7323.3574219, -9692.2382812, 9891.1328125

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8027376, upper bound: 7905.8345177
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8198602
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1627.6329346, 8015.9062500, -1617.2574463, 7942.2695312, -9569.9023438, 9633.1640625
1: -2533.1064453, 9285.1699219, -2516.1965332, 9201.5732422, -11734.6767578, 11801.3642578
2: -2197.8559570, 9580.0595703, -2184.7646484, 9491.6357422, -11689.4912109, 11764.8242188
3: -3381.9667969, 7036.8022461, -3361.2006836, 6981.9604492, -10363.9277344, 10398.0019531
4: -2334.1520996, 7491.1982422, -2320.7592773, 7428.2260742, -9762.3779297, 9811.9560547

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8025422, upper bound: 7905.8239804
time: 1.16 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8206700
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1651.7623291, 8134.5854492, -1617.2574463, 7942.2695312, -9594.0322266, 9751.8427734
1: -2570.5659180, 9423.0849609, -2516.1965332, 9201.5732422, -11772.1386719, 11939.2802734
2: -2230.6286621, 9721.8144531, -2184.7646484, 9491.6357422, -11722.2646484, 11906.5791016
3: -3433.0656738, 7142.9433594, -3361.2006836, 6981.9604492, -10415.0263672, 10504.1435547
4: -2368.8823242, 7603.1733398, -2320.7592773, 7428.2260742, -9797.1083984, 9923.9306641

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8025422, upper bound: 7905.8456147
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B2_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8373873
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1475.4304199, 7222.3227539, -1436.3679199, 7025.7211914, -8501.1513672, 8658.6904297
1: -2295.6276855, 8366.3876953, -2235.0996094, 8138.2026367, -10433.8271484, 10601.4873047
2: -1995.1540527, 8634.6416016, -1942.3479004, 8401.1142578, -10396.2666016, 10576.9882812
3: -3070.7192383, 6355.2846680, -2989.4938965, 6181.4941406, -9252.2119141, 9344.7773438
4: -2127.2221680, 6765.2705078, -2072.4086914, 6582.4228516, -8709.6445312, 8837.6796875

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
time: 0.88 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1502.6279297, 7360.3056641, -1425.8928223, 6974.2841797, -8476.9121094, 8786.1982422
1: -2338.5095215, 8526.4140625, -2219.0373535, 8078.3886719, -10416.8984375, 10745.4501953
2: -2031.7536621, 8799.7539062, -1928.2445068, 8339.0771484, -10370.8310547, 10727.9980469
3: -3129.4514160, 6476.4692383, -2966.3820801, 6134.8134766, -9264.2617188, 9442.8515625
4: -2167.3234863, 6894.8305664, -2056.0476074, 6532.8500977, -8700.1738281, 8950.8779297

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -1449.1295166, 7089.1728516, -1474.8061523, 7215.9350586, -8665.0644531, 8563.9785156
1: -2254.9724121, 8211.5361328, -2294.7985840, 8358.6699219, -10613.6425781, 10506.3349609
2: -1959.5747070, 8476.6738281, -1994.2857666, 8627.8652344, -10587.4394531, 10470.9589844
3: -3015.5468750, 6236.4956055, -3069.1271973, 6348.9091797, -9364.4550781, 9305.6220703
4: -2090.3281250, 6641.1000977, -2126.7553711, 6759.8911133, -8850.2177734, 8767.8544922

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8372383, upper bound: 7905.8196586
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8014170, upper bound: 7905.8088823
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1486.8823242, 7276.1162109, -1474.8061523, 7215.9350586, -8702.8173828, 8750.9208984
1: -2313.3249512, 8428.4785156, -2294.7985840, 8358.6699219, -10671.9941406, 10723.2773438
2: -2010.2305908, 8699.4619141, -1994.2857666, 8627.8652344, -10638.0957031, 10693.7460938
3: -3094.1743164, 6402.1845703, -3069.1271973, 6348.9091797, -9443.0800781, 9471.3105469
4: -2144.3576660, 6816.0834961, -2126.7553711, 6759.8911133, -8904.2490234, 8942.8388672

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8372383, upper bound: 7905.8196586
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8014170, upper bound: 7905.8088823
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1460.5506592, 7147.2270508, -1454.2230225, 7113.0654297, -8573.6152344, 8601.4492188
1: -2272.3906250, 8279.2373047, -2262.7141113, 8239.5361328, -10511.9267578, 10541.9501953
2: -1975.1887207, 8544.5400391, -1966.5577393, 8504.8789062, -10480.0664062, 10511.0976562
3: -3038.8251953, 6288.2265625, -3026.0964355, 6258.4028320, -9297.2285156, 9314.3232422
4: -2105.1496582, 6694.4780273, -2097.0598145, 6663.8061523, -8768.9560547, 8791.5380859

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8314799, upper bound: 7905.8342584
time: 0.73 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8314799, upper bound: 7905.8342584
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1449.1295166, 7089.1728516, -1486.7606201, 7275.5112305, -8724.6406250, 8575.9326172
1: -2254.9724121, 8211.5361328, -2313.1350098, 8427.7792969, -10682.7519531, 10524.6708984
2: -1959.5747070, 8476.6738281, -2010.0672607, 8698.7373047, -10658.3085938, 10486.7412109
3: -3015.5468750, 6236.4956055, -3093.9216309, 6401.6611328, -9417.2070312, 9330.4160156
4: -2090.3281250, 6641.1000977, -2144.1887207, 6815.5258789, -8905.8525391, 8785.2861328

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8329094, upper bound: 7905.8052303
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8033777, upper bound: 7905.8002297
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1474.9683838, 7216.7451172, -1486.7606201, 7275.5112305, -8750.4794922, 8703.5039062
1: -2295.0522461, 8359.6025391, -2313.1350098, 8427.7792969, -10722.8310547, 10672.7373047
2: -1994.5032959, 8628.8320312, -2010.0672607, 8698.7373047, -10693.2392578, 10638.8994141
3: -3069.4631348, 6349.6079102, -3093.9216309, 6401.6611328, -9471.1230469, 9443.5292969
4: -2126.9819336, 6760.6386719, -2144.1887207, 6815.5258789, -8942.5078125, 8904.8251953

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9019321, upper bound: 7905.8886726
time: 0.80 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8916569, upper bound: 7905.8886758
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1449.1295166, 7089.1728516, -1507.2565918, 7372.9223633, -8822.0517578, 8596.4296875
1: -2254.9724121, 8211.5361328, -2344.7475586, 8541.3300781, -10796.3027344, 10556.2832031
2: -1959.5747070, 8476.6738281, -2037.9860840, 8815.7333984, -10775.3056641, 10514.6591797
3: -3015.5468750, 6236.4956055, -3137.9196777, 6491.0332031, -9506.5791016, 9374.4140625
4: -2090.3281250, 6641.1000977, -2174.5458984, 6909.5263672, -8999.8544922, 8815.6425781

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8329094, upper bound: 7905.8362601
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8033775, upper bound: 7905.8334535
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1474.9683838, 7216.7451172, -1507.2565918, 7372.9223633, -8847.8906250, 8724.0019531
1: -2295.0522461, 8359.6025391, -2344.7475586, 8541.3300781, -10836.3818359, 10704.3496094
2: -1994.5032959, 8628.8320312, -2037.9860840, 8815.7333984, -10810.2363281, 10666.8183594
3: -3069.4631348, 6349.6079102, -3137.9196777, 6491.0332031, -9560.4951172, 9487.5273438
4: -2126.9819336, 6760.6386719, -2174.5458984, 6909.5263672, -9036.5078125, 8935.1816406

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8329094, upper bound: 7905.8315534
time: 1.01 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A1_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8033777, upper bound: 7905.8261641
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1486.8823242, 7276.1162109, -1486.7606201, 7275.5112305, -8762.3935547, 8762.8750000
1: -2313.3249512, 8428.4785156, -2313.1350098, 8427.7792969, -10741.1044922, 10741.6132812
2: -2010.2305908, 8699.4619141, -2010.0672607, 8698.7373047, -10708.9667969, 10709.5283203
3: -3094.1743164, 6402.1845703, -3093.9216309, 6401.6611328, -9495.8330078, 9496.1044922
4: -2144.3576660, 6816.0834961, -2144.1887207, 6815.5258789, -8959.8837891, 8960.2714844

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8277390, upper bound: 7905.8037251
time: 0.83 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7971199, upper bound: 7905.7982465
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1507.4025879, 7373.6542969, -1486.7606201, 7275.5112305, -8782.9140625, 8860.4121094
1: -2344.9753418, 8542.1738281, -2313.1350098, 8427.7792969, -10772.7529297, 10855.3076172
2: -2038.1811523, 8816.6074219, -2010.0672607, 8698.7373047, -10736.9169922, 10826.6738281
3: -3138.2192383, 6491.6611328, -3093.9216309, 6401.6611328, -9539.8789062, 9585.5791016
4: -2174.7465820, 6910.1982422, -2144.1887207, 6815.5258789, -8990.2714844, 9054.3847656

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B1_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9035156, upper bound: 7905.9023386
time: 0.86 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B1_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.9022866, upper bound: 7905.9021977
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1486.8823242, 7276.1162109, -1507.2565918, 7372.9223633, -8859.8046875, 8783.3730469
1: -2313.3249512, 8428.4785156, -2344.7475586, 8541.3300781, -10854.6552734, 10773.2246094
2: -2010.2305908, 8699.4619141, -2037.9860840, 8815.7333984, -10825.9628906, 10737.4453125
3: -3094.1743164, 6402.1845703, -3137.9196777, 6491.0332031, -9585.2050781, 9540.1025391
4: -2144.3576660, 6816.0834961, -2174.5458984, 6909.5263672, -9053.8837891, 8990.6289062

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8277390, upper bound: 7905.8142568
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7971199, upper bound: 7905.8052797
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_B2_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1507.4025879, 7373.6542969, -1507.2565918, 7372.9223633, -8880.3251953, 8880.9101562
1: -2344.9753418, 8542.1738281, -2344.7475586, 8541.3300781, -10886.3027344, 10886.9189453
2: -2038.1811523, 8816.6074219, -2037.9860840, 8815.7333984, -10853.9140625, 10854.5927734
3: -3138.2192383, 6491.6611328, -3137.9196777, 6491.0332031, -9629.2509766, 9629.5771484
4: -2174.7465820, 6910.1982422, -2174.5458984, 6909.5263672, -9084.2734375, 9084.7412109

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.8277390, upper bound: 7905.8298990
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_B2_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -7905.7971199, upper bound: 7905.8238931
time: 0.74 seconds

## Summary of splitting at layer (split count: 9)
- Time for NS candidates: 2.60 seconds
NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8504414
NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8867698, upper bound: 7905.8908363
NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8916187, upper bound: 7905.8771485
NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8768044
NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8916187, upper bound: 7905.8772666
NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8769095
NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8717704, upper bound: 7905.8577448
NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8453412, upper bound: 7905.8475581
NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8717704, upper bound: 7905.8658765
NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8453412, upper bound: 7905.8571317
NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8668871, upper bound: 7905.8477187
NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8405464, upper bound: 7905.8463065
NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8967401, upper bound: 7905.8858897
NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8856234
NS_A2_B2_B2_A1_B1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8668871, upper bound: 7905.8568827
NS_A2_B2_B2_A1_B1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8405464, upper bound: 7905.8568825
NS_A2_B2_B2_A1_B1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8967400, upper bound: 7905.8887678
NS_A2_B2_B2_A1_B1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8886638
NS_A2_B2_B2_A1_B1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8634242, upper bound: 7905.8459574
NS_A2_B2_B2_A1_B1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8359456, upper bound: 7905.8412120
NS_A2_B2_B2_A1_B1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8964908, upper bound: 7905.8853247
NS_A2_B2_B2_A1_B1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8834538, upper bound: 7905.8849334
NS_A2_B2_B2_A1_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8634242, upper bound: 7905.8505421
NS_A2_B2_B2_A1_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8359456, upper bound: 7905.8445019
NS_A2_B2_B2_A1_B1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8634242, upper bound: 7905.8546179
NS_A2_B2_B2_A1_B1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8359456, upper bound: 7905.8507376
NS_A2_B2_B2_A1_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
NS_A2_B2_B2_A1_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
NS_A2_B2_B2_A1_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
NS_A2_B2_B2_A1_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8903732, upper bound: 7905.8979838
NS_A2_B2_B2_A1_B2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.8597299
NS_A2_B2_B2_A1_B2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.9096967
NS_A2_B2_B2_A1_B2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.8597299
NS_A2_B2_B2_A1_B2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8534572, upper bound: 7905.9096967
NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8496093, upper bound: 7905.8398724
NS_A2_B2_B2_A1_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8495997, upper bound: 7905.8399763
NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_B1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8384038, upper bound: 7905.8291458
NS_A2_B2_B2_A1_B2_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8583933, upper bound: 7905.8434199
NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8489204, upper bound: 7905.8326640
NS_A2_B2_B2_A1_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8492876, upper bound: 7905.8327884
NS_A2_B2_B2_A1_B2_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8276262, upper bound: 7905.8555007
NS_A2_B2_B2_A1_B2_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8276262, upper bound: 7905.8572729
NS_A2_B2_B2_A1_B2_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8476173, upper bound: 7905.8702402
NS_A2_B2_B2_A1_B2_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8476173, upper bound: 7905.8702402
NS_A2_B2_B2_A1_B2_A2_B2_B1_B2_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8234755, upper bound: 7905.8120729
NS_A2_B2_B2_A1_B2_A2_B2_B1_B2_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8234755, upper bound: 7905.8347058
NS_A2_B2_B2_A2_B1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9056257, upper bound: 7905.9045529
NS_A2_B2_B2_A2_B1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9056257, upper bound: 7905.9059293
NS_A2_B2_B2_A2_B1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9037655, upper bound: 7905.9004054
NS_A2_B2_B2_A2_B1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9037655, upper bound: 7905.9009007
NS_A2_B2_B2_A2_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9094449, upper bound: 7905.8937816
NS_A2_B2_B2_A2_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8987859, upper bound: 7905.8941269
NS_A2_B2_B2_A2_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9051740, upper bound: 7905.9062771
NS_A2_B2_B2_A2_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9037655, upper bound: 7905.9039687
NS_A2_B2_B2_A2_B1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7997300, upper bound: 7905.8268131
NS_A2_B2_B2_A2_B1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7924470, upper bound: 7905.8151094
NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
NS_A2_B2_B2_A2_B1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
NS_A2_B2_B2_A2_B1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8529875, upper bound: 7905.8645076
NS_A2_B2_B2_A2_B1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8529875, upper bound: 7905.8645073
NS_A2_B2_B2_A2_B1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8720209
NS_A2_B2_B2_A2_B1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8561632, upper bound: 7905.8717642
NS_A2_B2_B2_A2_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8431974
NS_A2_B2_B2_A2_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8541876
NS_A2_B2_B2_A2_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8431974
NS_A2_B2_B2_A2_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8600168, upper bound: 7905.8607735
NS_A2_B2_B2_A2_B1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8027376, upper bound: 7905.8268185
NS_A2_B2_B2_A2_B1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8177655
NS_A2_B2_B2_A2_B1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8027376, upper bound: 7905.8345177
NS_A2_B2_B2_A2_B1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8198602
NS_A2_B2_B2_A2_B1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8025422, upper bound: 7905.8239804
NS_A2_B2_B2_A2_B1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8206700
NS_A2_B2_B2_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8025422, upper bound: 7905.8456147
NS_A2_B2_B2_A2_B1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7997610, upper bound: 7905.8373873
NS_A2_B2_B2_A2_B2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
NS_A2_B2_B2_A2_B2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
NS_A2_B2_B2_A2_B2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
NS_A2_B2_B2_A2_B2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8930394, upper bound: 7905.8937804
NS_A2_B2_B2_A2_B2_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8372383, upper bound: 7905.8196586
NS_A2_B2_B2_A2_B2_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8014170, upper bound: 7905.8088823
NS_A2_B2_B2_A2_B2_B1_B2_A1_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8372383, upper bound: 7905.8196586
NS_A2_B2_B2_A2_B2_B1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8014170, upper bound: 7905.8088823
NS_A2_B2_B2_A2_B2_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8314799, upper bound: 7905.8342584
NS_A2_B2_B2_A2_B2_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8314799, upper bound: 7905.8342584
NS_A2_B2_B2_A2_B2_B2_A1_B1_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8329094, upper bound: 7905.8052303
NS_A2_B2_B2_A2_B2_B2_A1_B1_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8033777, upper bound: 7905.8002297
NS_A2_B2_B2_A2_B2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9019321, upper bound: 7905.8886726
NS_A2_B2_B2_A2_B2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8916569, upper bound: 7905.8886758
NS_A2_B2_B2_A2_B2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8329094, upper bound: 7905.8362601
NS_A2_B2_B2_A2_B2_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8033775, upper bound: 7905.8334535
NS_A2_B2_B2_A2_B2_B2_A1_B2_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8329094, upper bound: 7905.8315534
NS_A2_B2_B2_A2_B2_B2_A1_B2_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8033777, upper bound: 7905.8261641
NS_A2_B2_B2_A2_B2_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8277390, upper bound: 7905.8037251
NS_A2_B2_B2_A2_B2_B2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7971199, upper bound: 7905.7982465
NS_A2_B2_B2_A2_B2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9035156, upper bound: 7905.9023386
NS_A2_B2_B2_A2_B2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.9022866, upper bound: 7905.9021977
NS_A2_B2_B2_A2_B2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8277390, upper bound: 7905.8142568
NS_A2_B2_B2_A2_B2_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7971199, upper bound: 7905.8052797
NS_A2_B2_B2_A2_B2_B2_A2_B2_A2_A1, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.8277390, upper bound: 7905.8298990
NS_A2_B2_B2_A2_B2_B2_A2_B2_A2_A2, status: Status.VERIFIED, split count: 10, time: 2.60
Output dim: 3, lower bound: -7905.7971199, upper bound: 7905.8238931

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -1299.5012207, 6353.3828125, -1279.2634277, 6249.6245117, -7549.1259766, 7632.6464844
1: -2021.0819092, 7359.1831055, -1989.7108154, 7238.9174805, -9259.9970703, 9348.8935547
2: -1755.3535156, 7592.6337891, -1728.0113525, 7469.7480469, -9225.1015625, 9320.6445312
3: -2693.4650879, 5586.3168945, -2651.7514648, 5495.3662109, -8188.8310547, 8238.0683594
4: -1860.1925049, 5943.2314453, -1832.1188965, 5847.6782227, -7707.8696289, 7775.3496094

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8722054, upper bound: 7905.8609023
time: 0.81 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8696548, upper bound: 7905.8585605
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1299.5012207, 6353.3828125, -1572.0181885, 7715.5107422, -9015.0117188, 7925.4008789
1: -2021.0819092, 7359.1831055, -2446.0532227, 8938.3847656, -10959.4648438, 9805.2363281
2: -1755.3535156, 7592.6337891, -2123.4790039, 9221.7919922, -10977.1435547, 9716.1113281
3: -2693.4650879, 5586.3168945, -3266.8515625, 6781.4169922, -9474.8818359, 8853.1679688
4: -1860.1925049, 5943.2314453, -2256.8728027, 7216.7954102, -9076.9863281, 8200.1044922

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8578907, upper bound: 7905.8585605
time: 0.71 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8696548, upper bound: 7905.8585605
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -1591.8702393, 7817.7373047, -1279.2634277, 6249.6245117, -7841.4946289, 9096.9980469
1: -2476.7712402, 9056.9365234, -1989.7108154, 7238.9174805, -9715.6875000, 11046.6474609
2: -2150.2719727, 9342.8144531, -1728.0113525, 7469.7480469, -9620.0195312, 11070.8261719
3: -3307.7036133, 6870.8745117, -2651.7514648, 5495.3662109, -8803.0693359, 9522.6259766
4: -2284.3007812, 7310.7485352, -1832.1188965, 5847.6782227, -8131.9790039, 9142.8671875

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8445955
time: 0.78 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8446267, upper bound: 7905.8483257
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1591.8702393, 7817.7373047, -1572.0181885, 7715.5107422, -9307.3808594, 9389.7529297
1: -2476.7712402, 9056.9365234, -2446.0532227, 8938.3847656, -11415.1562500, 11502.9902344
2: -2150.2719727, 9342.8144531, -2123.4790039, 9221.7919922, -11372.0634766, 11466.2929688
3: -3307.7036133, 6870.8745117, -3266.8515625, 6781.4169922, -10089.1210938, 10137.7265625
4: -2284.3007812, 7310.7485352, -2256.8728027, 7216.7954102, -9501.0947266, 9567.6210938

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8453464, upper bound: 7905.8445955
time: 1.03 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8446267, upper bound: 7905.8483257
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1340.1763916, 6555.7343750, -1297.0157471, 6340.4389648, -7680.6142578, 7852.7500000
1: -2084.3928223, 7593.6889648, -2017.4185791, 7343.8872070, -9428.2802734, 9611.1064453
2: -1810.2260742, 7834.2006836, -1751.8430176, 7577.9272461, -9388.1533203, 9586.0429688
3: -2778.3894043, 5764.7529297, -2688.6369629, 5574.4418945, -8352.8310547, 8453.3896484
4: -1918.2866211, 6132.3623047, -1857.4781494, 5931.5810547, -7849.8666992, 7989.8388672

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8459915, upper bound: 7905.8559942
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8441024, upper bound: 7905.8474083
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1340.1763916, 6555.7343750, -1325.3457031, 6479.2104492, -7819.3862305, 7881.0800781
1: -2084.3928223, 7593.6889648, -2062.0219727, 7505.3334961, -9589.7255859, 9655.7099609
2: -1810.2260742, 7834.2006836, -1790.2436523, 7745.0410156, -9555.2666016, 9624.4443359
3: -2778.3894043, 5764.7529297, -2751.0649414, 5699.3012695, -8477.6904297, 8515.8183594
4: -1918.2866211, 6132.3623047, -1900.4581299, 6064.8725586, -7983.1582031, 8032.8203125

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8459915, upper bound: 7905.8559942
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8441024, upper bound: 7905.8474083
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1369.0230713, 6697.4321289, -1297.0157471, 6340.4389648, -7709.4619141, 7994.4477539
1: -2129.7778320, 7758.4409180, -2017.4185791, 7343.8872070, -9473.6650391, 9775.8574219
2: -1849.3094482, 8004.6962891, -1751.8430176, 7577.9272461, -9427.2363281, 9756.5390625
3: -2841.9899902, 5892.2075195, -2688.6369629, 5574.4418945, -8416.4316406, 8580.8447266
4: -1962.1009521, 6268.3403320, -1857.4781494, 5931.5810547, -7893.6816406, 8125.8178711

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1_A1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8614883, upper bound: 7905.8626460
time: 0.72 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1_A2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8417383, upper bound: 7905.8524312
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1369.0230713, 6697.4321289, -1325.3457031, 6479.2104492, -7848.2333984, 8022.7778320
1: -2129.7778320, 7758.4409180, -2062.0219727, 7505.3334961, -9635.1103516, 9820.4619141
2: -1849.3094482, 8004.6962891, -1790.2436523, 7745.0410156, -9594.3505859, 9794.9394531
3: -2841.9899902, 5892.2075195, -2751.0649414, 5699.3012695, -8541.2900391, 8643.2724609
4: -1962.1009521, 6268.3403320, -1900.4581299, 6064.8725586, -8026.9731445, 8168.7983398

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8454055, upper bound: 7905.8657934
time: 0.84 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8417383, upper bound: 7905.8524312
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -1279.4038086, 6250.3139648, -1317.2629395, 6437.6416016, -7717.0444336, 7567.5771484
1: -1989.9300537, 7239.7148438, -2048.8330078, 7456.8833008, -9446.8125000, 9288.5478516
2: -1728.2001953, 7470.5712891, -1779.3218994, 7694.2675781, -9422.4667969, 9249.8935547
3: -2652.0480957, 5495.9697266, -2730.9938965, 5661.3183594, -8313.3662109, 8226.9619141
4: -1832.3200684, 5848.3222656, -1886.3032227, 6023.6933594, -7856.0131836, 7734.6254883

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8804126, upper bound: 7905.8768146
time: 0.76 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8804126, upper bound: 7905.8768146
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -1572.1594238, 7716.2001953, -1296.6973877, 6346.2260742, -7918.3852539, 9012.8974609
1: -2446.2739258, 8939.1806641, -2017.2939453, 7350.5283203, -9796.8017578, 10956.4746094
2: -2123.6689453, 9222.6162109, -1751.6835938, 7583.4150391, -9707.0839844, 10974.2998047
3: -3267.1499023, 6782.0219727, -2687.8591309, 5576.6616211, -8843.8115234, 9469.8808594
4: -2257.0759277, 7217.4394531, -1855.7879639, 5934.0410156, -8191.1162109, 9073.2275391

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8804126, upper bound: 7905.8768146
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8804126, upper bound: 7905.8768146
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -1303.3890381, 6367.8071289, -1317.2629395, 6437.6416016, -7741.0307617, 7685.0703125
1: -2027.2319336, 7376.1289062, -2048.8330078, 7456.8833008, -9484.1142578, 9424.9619141
2: -1760.6683350, 7611.0566406, -1779.3218994, 7694.2675781, -9454.9355469, 9390.3779297
3: -2702.4460449, 5600.5571289, -2730.9938965, 5661.3183594, -8363.7646484, 8331.5507812
4: -1866.6339111, 5959.0800781, -1886.3032227, 6023.6933594, -7890.3271484, 7845.3833008

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8769095
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8769095
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -1595.2266846, 7828.6416016, -1296.6973877, 6346.2260742, -7941.4526367, 9125.3378906
1: -2482.1103516, 9069.7636719, -2017.2939453, 7350.5283203, -9832.6357422, 11087.0576172
2: -2154.9985352, 9357.1308594, -1751.6835938, 7583.4150391, -9738.4140625, 11108.8144531
3: -3315.8081055, 6882.4262695, -2687.8591309, 5576.6616211, -8892.4697266, 9570.2851562
4: -2290.2861328, 7323.7739258, -1855.7879639, 5934.0410156, -8224.3271484, 9179.5615234

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8769095
time: 0.82 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8751589, upper bound: 7905.8769095
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -1314.8543701, 6425.1762695, -1317.2629395, 6437.6416016, -7752.4960938, 7742.4394531
1: -2044.8015137, 7442.6376953, -2048.8330078, 7456.8833008, -9501.6835938, 9491.4697266
2: -1775.8753662, 7679.3237305, -1779.3218994, 7694.2675781, -9470.1425781, 9458.6455078
3: -2726.1525879, 5651.2631836, -2730.9938965, 5661.3183594, -8387.4707031, 8382.2539062
4: -1883.5876465, 6012.4907227, -1886.3032227, 6023.6933594, -7907.2812500, 7898.7939453

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8463044, upper bound: 7905.8475581
time: 0.74 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8463044, upper bound: 7905.8475581
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -1607.9882812, 7892.5844727, -1296.6973877, 6346.2260742, -7954.2143555, 9189.2802734
1: -2501.6855469, 9143.7236328, -2017.2939453, 7350.5283203, -9852.2119141, 11161.0156250
2: -2171.8012695, 9432.8212891, -1751.6835938, 7583.4150391, -9755.2167969, 11184.5048828
3: -3341.4904785, 6938.2607422, -2687.8591309, 5576.6616211, -8918.1523438, 9626.1201172
4: -2308.3322754, 7382.6298828, -1855.7879639, 5934.0410156, -8242.3730469, 9238.4160156

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8463044, upper bound: 7905.8475581
time: 0.85 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8463044, upper bound: 7905.8475581
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -1335.0505371, 6521.4287109, -1317.2629395, 6437.6416016, -7772.6923828, 7838.6914062
1: -2076.0173340, 7554.8803711, -2048.8330078, 7456.8833008, -9532.9003906, 9603.7128906
2: -1803.3531494, 7795.0068359, -1779.3218994, 7694.2675781, -9497.6201172, 9574.3291016
3: -2769.6354980, 5739.5502930, -2730.9938965, 5661.3183594, -8430.9531250, 8470.5439453
4: -1913.5245361, 6105.4580078, -1886.3032227, 6023.6933594, -7937.2172852, 7991.7612305

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561997, upper bound: 7905.8571317
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561997, upper bound: 7905.8571317
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -1628.0570068, 7987.8486328, -1296.6973877, 6346.2260742, -7974.2832031, 9284.5458984
1: -2532.7253418, 9254.8525391, -2017.2939453, 7350.5283203, -9883.2509766, 11272.1455078
2: -2199.2392578, 9547.4892578, -1751.6835938, 7583.4150391, -9782.6542969, 11299.1728516
3: -3384.9653320, 7025.9311523, -2687.8591309, 5576.6616211, -8961.6259766, 9713.7900391
4: -2338.2912598, 7474.9882812, -1855.7879639, 5934.0410156, -8272.3320312, 9330.7753906

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 33

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561997, upper bound: 7905.8571317
time: 0.92 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8561997, upper bound: 7905.8571317
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1279.4038086, 6250.3139648, -1328.6783447, 6494.4702148, -7773.8735352, 7578.9921875
1: -1989.9300537, 7239.7148438, -2066.3291016, 7522.7690430, -9512.6992188, 9306.0439453
2: -1728.2001953, 7470.5712891, -1794.4418945, 7761.9228516, -9490.1210938, 9265.0136719
3: -2652.0480957, 5495.9697266, -2754.6601562, 5711.7026367, -8363.7509766, 8250.6298828
4: -1832.3200684, 5848.3222656, -1903.2484131, 6076.7460938, -7909.0654297, 7751.5698242

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8591739, upper bound: 7905.8466218
time: 0.77 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8591739, upper bound: 7905.8466218
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1572.1594238, 7716.2001953, -1307.7850342, 6400.7070312, -7972.8662109, 9023.9853516
1: -2446.2739258, 8939.1806641, -2034.2434082, 7413.7626953, -9860.0371094, 10973.4238281
2: -2123.6689453, 9222.6162109, -1766.2982178, 7648.3076172, -9771.9765625, 10988.9140625
3: -3267.1499023, 6782.0219727, -2710.6086426, 5625.1420898, -8892.2919922, 9492.6308594
4: -2257.0759277, 7217.4394531, -1872.0865479, 5985.0068359, -8242.0830078, 9089.5263672

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8591739, upper bound: 7905.8466218
time: 0.79 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8591739, upper bound: 7905.8466218
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1320.5654297, 6455.0498047, -1337.0828857, 6537.4184570, -7857.9833984, 7792.1328125
1: -2054.0061035, 7476.9565430, -2079.4660645, 7572.4228516, -9626.4277344, 9556.4228516
2: -1783.7268066, 7714.9448242, -1805.7202148, 7813.1625977, -9596.8886719, 9520.6640625
3: -2737.8513184, 5676.3750000, -2772.3093262, 5749.2172852, -8487.0664062, 8448.6835938
4: -1891.0191650, 6039.5585938, -1915.4144287, 6116.5781250, -8007.5971680, 7954.9731445

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 25

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8856134
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8856134
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1349.3181152, 6595.9804688, -1329.9431152, 6501.1538086, -7850.4716797, 7925.9233398
1: -2099.2395020, 7640.9106445, -2068.6062012, 7530.5063477, -9629.7460938, 9709.5166016
2: -1822.6904297, 7884.5996094, -1796.2683105, 7769.8925781, -9592.5830078, 9680.8681641
3: -2801.2390137, 5803.2675781, -2757.4296875, 5717.4873047, -8518.7265625, 8560.6972656
4: -1934.7098389, 6174.9272461, -1905.0789795, 6082.8437500, -8017.5517578, 8080.0063477

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8856234
time: 0.75 seconds

## Relational analysis of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B2_B2_A1_B1_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -7905.8842181, upper bound: 7905.8856234
time: 0.88 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.98 + 418.12 = 421.09 seconds
